import os
import sys
sys.path.append('../')
import numpy as np
import torch
import random
import decord
from decord import VideoReader, cpu
from PIL import Image
from pprint import pprint
from tqdm import tqdm 
import subprocess
import time
import argparse

import json
import re
import multiprocessing
from multiprocessing import Pool, Manager, current_process
from functools import partial

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info 


def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)


def init_model_for_process(gpu_ids, args):
    global model, processor, tokenizer

    process_id = int(current_process().name.split('-')[-1])-1  # 提取进程编号 0,1,2,3,...,15

    gpu_id = process_id % len(gpu_ids)      # should be 0,1,2,...,7,0,1,2,...,7
    
    print(f"进程 {process_id} 加载到GPU {gpu_id}")
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # 绑定 GPU
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.vl_model,  # "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,       # torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=None,    # close auto
        )
    model.to("cuda:"+str(gpu_id))
    processor = AutoProcessor.from_pretrained(args.vl_model)
    tokenizer = AutoTokenizer.from_pretrained(args.vl_model)


def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""




def generate_masks(n, num_masks=10, ones_per_mask=16, mode='both'):
    """
    mode: 'block' 连续块, 'stride' 等步长, 'both' 随机选择
    """
    if ones_per_mask >= n:
        print(f"Warning: ones_per_mask ({ones_per_mask}) >= n ({n}), set ones_per_mask = n // 2 = {n // 2}")
        ones_per_mask = n // 2
    
    masks = []
    for _ in range(num_masks):
        if mode == 'both':
            this_mode = random.choice(['block', 'stride'])
        else:
            this_mode = mode

        mask = torch.zeros(n, dtype=torch.int)
        if this_mode == 'block':
            # 连续块
            if n == ones_per_mask:
                start = 0
            else:
                start = random.randint(0, n - ones_per_mask)
            mask[start:start+ones_per_mask] = 1
        elif this_mode == 'stride':
            # 等步长
            max_stride = (n - 1) // (ones_per_mask - 1) if ones_per_mask > 1 else n
            stride = random.randint(1, max_stride)
            start = random.randint(0, n - (ones_per_mask - 1) * stride - 1)
            for i in range(ones_per_mask):
                mask[start + i * stride] = 1
        else:
            raise ValueError('Unknown mode')
        masks.append(mask)
    # add a all 1
    masks.append(torch.ones(n, dtype=torch.int))
    return torch.stack(masks)



def process_video(video_path, args):
    global model, processor, tokenizer
    (video_path, gpu_id, args) = video_path
    
    # basic info
    video_name = video_path['path'].split('/')[-1]      # ytb_7nRmsEw7nsE.mp4
    data_root = os.environ.get("VIDEO_R1_DATA_ROOT", "./Video-R1-data")
    v_path = os.path.join(data_root, video_path['path'].lstrip('/'))
    video_file = str(video_path['problem_id']) + '.json'   # xxxxx.json
    
    gt_answer = extract_answer(video_path['solution'])
    if video_path['problem_type'] == 'multiple choice':
        for option in video_path['options']:
            if option.startswith(gt_answer):
                gt_answer = option[3:]
    
    question = video_path['problem']

    full_input = question + " " + gt_answer  

    if os.path.exists(os.path.join(args.output_base_path, video_file)):     # do not process again
        print(f"{video_path} has already been processed.")
        return video_path, None
    
    try:
        messages_prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": v_path,
                        "max_pixels": 128 * 256,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        messages_full = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": v_path,
                        "max_pixels": 128 * 256,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": full_input},
                ],
            }
        ]

        text_prompt = processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        text_full = processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=True
        )
        # print(text_prompt, '\n', text_full)

        image_inputs, video_inputs = process_vision_info(messages_full)      # [(n, 3, h, w)]
        generated_mask = generate_masks(video_inputs[0].shape[0], args.num_masks, args.ones_per_mask, 'both')     # 10 * n
        mask_loss = []


        # iterate mask
        for (i, mask) in enumerate(generated_mask):
            selected_frames = video_inputs[0][mask==1]     # 16 3 H W?
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=[selected_frames],
                padding=True,
                return_tensors="pt",
            )
            prompt_len = inputs['input_ids'].shape[1] - 1      # 减1因为BOS
            del inputs

            model_inputs = processor(
                text=[text_full],
                images=image_inputs,
                videos=[selected_frames],       # TODO: a bug here, loss只和分辨率有关系，和帧内容无关
                padding=True,
                return_tensors="pt",
                padding_side='left'
            ).to(model.device)
            input_ids = model_inputs["input_ids"]   # [1, seq_len]
            attention_mask = model_inputs["attention_mask"]
            labels = input_ids.clone().to(model.device)
            labels[:, :prompt_len] = -100   

            with torch.no_grad():
                loss = model(input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    pixel_values_videos=model_inputs["pixel_values_videos"],
                    video_grid_thw=model_inputs["video_grid_thw"],           
                    second_per_grid_ts=model_inputs["second_per_grid_ts"],   
                    labels=labels).loss

            mask_loss.append({
                "mask": str(mask.tolist()),
                "loss": loss.item(),
            })
            
    except Exception as e:
        print(e)
        print(f"error when processing video {video_path['path']}")
        return video_path, None
        
    # save each json
    save_json(mask_loss, os.path.join(args.output_base_path, video_file))
    print(f"{video_file} is processed.")
    torch.cuda.empty_cache()
    return video_path, None



def vl_inference(args, video_list, gpu_ids ):
    if not os.path.exists(args.output_base_path):
        os.makedirs(args.output_base_path)

    num_processes = args.num_processes

    tasks = [(video_dict, gpu_ids[i % len(gpu_ids)], args) for i, video_dict in enumerate(video_list)]
    partial_process_video = partial(process_video, args=args)

    with Pool(processes=num_processes, initializer=init_model_for_process, initargs=(gpu_ids, args)) as pool:
        for result in pool.imap_unordered(partial_process_video, tasks):
            # print(f"{result[0]} has been processed successfully")
            pass

    return None
   

def get_visible_gpus():
    """获取环境变量中设置的 CUDA_VISIBLE_DEVICES"""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices:
        # 返回一个列表，其中每个元素是逻辑 GPU ID
        return [int(x) for x in visible_devices.split(",")]
    else:
        # 如果未设置，默认使用所有物理 GPU
        from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlShutdown
        nvmlInit()
        num_gpus = nvmlDeviceGetCount()
        nvmlShutdown()
        return list(range(num_gpus))  # 返回所有物理 GPU ID



def filter_data(data):
    problem_ids = []
    with open('../used_data_5cards.txt', 'r') as f:
        for line in f:
            if 'problem_id:' in line:
                pid = line.strip().split('problem_id:')[-1].strip()
                problem_ids.append(int(pid))

    with open('../used_training_data.txt', 'r') as f:
        for line in f:
            if 'problem_id:' in line:
                pid = line.strip().split('problem_id:')[-1].strip()
                problem_ids.append(int(pid))
    problem_ids = list(set(problem_ids))

    video_data = []
    for item in data:
        if item['data_type'] == 'video' and item['problem_id'] not in problem_ids:
            video_data.append(item)
    return video_data


def main(video_list):
    multiprocessing.set_start_method('spawn')
    
    # get gpu ids
    gpu_ids = get_visible_gpus()    # [0,1,2,3]

    args = parse_args()
    video_list = sorted(video_list, key=lambda x: x['problem_id'], reverse=False)       # 正向排
    # return output by feeding data to pipe
    all_captions = vl_inference(args, video_list, gpu_ids)
    output_path = os.path.join(args.output_base_path, args.output_filename)

    return None

def parse_args():
    parser = argparse.ArgumentParser("")
     # VL model
    parser.add_argument("--vl_model", default='/path/to/InternVL-Chat-V1-5', type=str) 
    parser.add_argument("--n_gpu", default=4, type=int)  # gpu used
    # output
    parser.add_argument("--output_base_path", default='output', type=str)  
    parser.add_argument("--output_filename", default='sad.json', type=str)  
    # other
    parser.add_argument("--num_processes", default=4, type=int)
    parser.add_argument("--num_masks", default=10, type=int)
    parser.add_argument("--ones_per_mask", default=16, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    with open("./Video-R1-260k.json", 'r') as f:
        all_data = json.load(f)     # len: 263071
    
    # filter out videos
    video_data = filter_data(all_data)      # len: 116248
    main(video_data)




