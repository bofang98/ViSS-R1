import os
import json
import re
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse

import multiprocessing as mp
import random


def evaluate_dataset(output_path_root, dataset_name, llm, sampling_params, processor, file_name, BSZ=64):
    
    # OUTPUT_PATH = f"./src/r1-v/eval_results/eval_{dataset_name}_{file_name}_greedy_output.json"

    if not os.path.exists(os.path.join(output_path_root, "eval_results_ssl")):
        os.makedirs(os.path.join(output_path_root, "eval_results_ssl"), exist_ok=True)

    OUTPUT_PATH = os.path.join(output_path_root, f"eval_results_ssl/{dataset_name}_{file_name}.json")
    PROMPT_PATH = f"./src/r1-v/Evaluation/eval_{dataset_name}.json"

    if not os.path.isfile(PROMPT_PATH):
        raise FileNotFoundError(f"[{dataset_name}] Prompt file not found: {PROMPT_PATH}")
        
    
    if PROMPT_PATH.endswith('.jsonl'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    elif PROMPT_PATH.endswith('.json'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Input file must be .json or .jsonl")


    # my adapted
    QUESTION_TEMPLATE = (
        "You will be given an image or video that has been transformed in some way, along with two progressive questions:\n"
        "1. Transformation Identification: Based on the visual content, identify the type of transformation applied to the image or video.\n"
        "2. User Question: On the basis of the restored (inverse-transformed) image or video, answer the real user's question. You need to think even more deeply about the user question.\n"
        "Please think through both questions step by step, as a human would. Carefully analyze the visual content and reason in detail. "
        "Your thought process should include natural inner dialogue such as 'let me think,' 'hmm,' 'wait,' 'oh, I see,' 'let’s break it down,' etc., and show self-reflection or verification as you work through each problem. "
        "Put your detailed thought process inside <think>...</think> tags. Place your final transformation identification inside <transform>...</transform> tags. Place your final answer to the user question inside <answer>...</answer> tags. For example: <think> ... </think><transform> ... </transform><answer> ... </answer>.\n"
        "Important:\n"
        "- Your reasoning and answer must be based on the visual information from the input image or video.\n"
        "- For answering the Transform Question, provide only the single option letter (e.g., A, B, etc.) within the <transform> </transform> tags.\n"
        "- For answering the User Question, {Type_template}\n"
        "Transform question: {Trans_question}\n"
        "User Question: {Question}"
    )

    TYPE_TEMPLATE = {
        "multiple choice": "provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": "provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": "transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": "provide your text answer within the <answer> </answer> tags.",
        "regression": "provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    TRANS_VIDEO_QUESTION = {
        "rotate": (
            "Please review the transformed video and determine which rotation angle was most likely applied to the original video. "
            "The possible options are:\n "
            "A. 0°\n B. 90°\n C. 180°\n D. 270°\n"
            "Note: The rotation is applied in the counterclockwise (anticlockwise) direction and 0° corresponds to no rotation.\n"
        ),
        "arrow": (
            "Please review the input video and determine whether the sequence of video frames has been reversed (i.e., played in reverse order). "
            "The possible options are:\n "
            "A. No, the video frames remain in their original order.\n "
            "B. Yes, the video frames have been reversed.\n"
        ),
        "shuffle": (
            "Please review the input video. The video originally consists of 32 frames, divided into 4 consecutive groups as follows:\n "
            "Group 1: Frames 1-8\n Group 2: Frames 9-16\n Group 3: Frames 17-24\n Group 4: Frames 25-32\n"
            "Now two of these groups have been randomly selected and swapped with each other, while the remaining groups stayed in their original positions. "
            "Please identify which two groups are swapped.\n "
            "A. Group 1 and Group 2 were swapped\n "
            "B. Group 1 and Group 3 were swapped\n "
            "C. Group 1 and Group 4 were swapped\n "
            "D. Group 2 and Group 3 were swapped\n "
            "E. Group 2 and Group 4 were swapped\n "
            "F. Group 3 and Group 4 were swapped\n"
        )
    }

    random_ssl_question = TRANS_VIDEO_QUESTION[random.choice(['rotate', 'arrow', 'shuffle'])]


    messages = []
    for x in data:
        if x["problem_type"] == 'multiple choice':
            question = x['problem'] + "Options:\n"
            for op in x["options"]:
                question += op + "\n"
        else:
            question = x['problem']

        msg = [{
            "role": "user",
            "content": [
                {
                    "type": x['data_type'],
                    # x['data_type']: os.getcwd() + "/src/r1-v/Evaluation" + x['path'][1:]
                    x['data_type']: "/gpubox03/ssd5/fangbo05/Video-R1" + x['path'][1:]
                },
                {
                    "type": "text",
                    # "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[x['problem_type']]
                    "text": QUESTION_TEMPLATE.format(Question=question, Type_template=TYPE_TEMPLATE[x['problem_type']], Trans_question=random_ssl_question)
                }
            ]
        }]
        messages.append(msg)

    final_output = []
    start_idx = 0
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
                final_output = existing.get("results", [])
                start_idx = len(final_output)
                print(f"[{dataset_name}] Resuming from sample index {start_idx}")
        except Exception as e:
            print(f"[{dataset_name}] Error reading existing output file: {e}")



    def extract_think(output_str):
        pattern = r'<think>\s*(.*?)\s*</think>'
        match = re.search(pattern, output_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            return None
        
    def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):

        if not torch.is_tensor(pred):
            pred = torch.tensor(pred, dtype=torch.float32)
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float32)
        
        epsilon = 1e-8
        rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
        
        thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
        
        conditions = rel_error < (1 - thresholds)  
        mra = conditions.float().mean()  
        return mra.item()


    def reward_fn(sample, model_output, question_type):
        try:
            output_ans = extract_answer(model_output)
            if output_ans == '':
                output_ans = model_output
            gt_ans = extract_answer(sample.get("solution", ""))
            if question_type == "multiple choice":
                return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    return 0.0
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                mra = mean_relative_accuracy(out_number, gt_number)
                return mra
            else:
                return 0.0
        except Exception as e:
            return 0.0

    


    mean_acc = []
    mean_mra = []
    for i in tqdm(range(start_idx, len(messages), BSZ), desc=f"[{dataset_name}] Processing"):
        batch_messages = messages[i:i + BSZ]
        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]

        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)

            image_idx = 0
            video_idx = 0
            llm_inputs = []

            for idx, prompt in enumerate(prompts):
                mm_type = batch_messages[idx][0]['content'][0]['type']
                sample_mm_data = {}
                sample_video_kw = {}
                if mm_type == 'image':
                    sample_mm_data["image"] = image_inputs[image_idx]
                    image_idx += 1
                elif mm_type == 'video':
                    sample_mm_data["video"] = video_inputs[video_idx]
                    for key, value in video_kwargs.items():
                        sample_video_kw[key] = value[video_idx]
                    video_idx += 1

                llm_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": sample_mm_data,
                    "mm_processor_kwargs": sample_video_kw,
                })

            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            batch_output_text = [out.outputs[0].text for out in outputs]

        except Exception as e:
            print(f'[{dataset_name}] error: {data[i]["path"] if i < len(data) else "index"}')
            print('Exception:', e)
            batch_output_text = ['<answer>error</answer>'] * len(batch_messages)

        for j, (sample, model_output) in enumerate(zip(data[i:i+BSZ], batch_output_text), start=i):
            think_chain = extract_think(model_output)
            final_ans = extract_answer(model_output)
            if final_ans == "":
                final_ans = model_output
            sample["output"] = model_output
            sample["prediction"] = final_ans
            q_type = sample.get("problem_type", "")
            sample["reward"] = reward_fn(sample, model_output, q_type)
            sample['correct'] = True if sample["reward"]==1.0 else False
            if sample['problem_type'] != 'regression':
                mean_acc.append(sample["reward"])
            else:
                mean_mra.append(sample["reward"])
            if think_chain:
                sample["process"] = f"<think>{think_chain}</think>"
            final_output.append(sample)

        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
            print(f"[{dataset_name}] Processed batch {(i - start_idx)//BSZ + 1}, saved {len(final_output)} samples.")
        except Exception as e:
            print(f"[{dataset_name}] Error writing to output file: {e}")

    final_acc={'mean_acc': 0.0, 'mean_mra': 0.0}
    if len(mean_acc) > 0:
        # if dataset_name == 'mvbench':
        #     # print(len(mean_acc))
        #     # print(mean_acc.sum())
        #     final_acc['mean_acc'] = mean_acc.sum() / 3800      # 3800 videos in mvbench
        # else:
        final_acc['mean_acc'] = torch.tensor(mean_acc).mean().item()
    if len(mean_mra) > 0:
        final_acc['mean_mra'] = torch.tensor(mean_mra).mean().item()
    
    if dataset_name == 'mvbench':
        final_acc['mean_acc'] = final_acc['mean_acc'] * 4000 / 3800

    if dataset_name == 'vsibench':
        final_acc['avg'] = (final_acc['mean_acc'] + final_acc['mean_mra'])/2
    
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": final_output, "final_acc": [final_acc]}, f, indent=2, ensure_ascii=False)
        print(f"[{dataset_name}] Final accuracy saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"[{dataset_name}] Error writing final accuracy to output file: {e}")

    print(f"[{dataset_name}] Results saved to {OUTPUT_PATH}")

    



def _worker(dataset_name, gpu_id, model_path, file_name, port):
    
    # 固定到单块 GPU（非常关键，必须在任何 CUDA 库被导入前设置）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    for key in ("MASTER_PORT", "TORCH_DIST_PORT", "VLLM_PORT"):
        os.environ[key] = str(port)

    # 现在再导入 CUDA 相关包
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor, AutoTokenizer

    # 构建 sampling_params（如需和主进程一致，可以直接在这里写死或按 args 传）
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        max_tokens=1024,
        stop_token_ids=[],
    )

    # 单卡实例：tensor_parallel_size 设为 1
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=8192 * 2,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 1, "video": 1},
    )

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    output_path_root = os.path.dirname(model_path)

    # 开始评测一个数据集
    evaluate_dataset(output_path_root, dataset_name, llm, sampling_params, processor, file_name)



def get_args():
    parser = argparse.ArgumentParser(description="Evaluation benchmark")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
    parser.add_argument('--file_name', type=str, required=True, help="Name of the file")
    parser.add_argument('--port_base', type=int, default=15100, help="Port base")
    parser.add_argument("--port_stride", type=int, default=10, help="port stride")
    
    return parser.parse_args()


def main():
    args = get_args()
    MODEL_PATH = args.model_path
    file_name = args.file_name

    # benchmarks
    datasets = ['tempcompass', 'vsibench', 'mvbench', 'videomme', 'videommmu', 'mmvu']


    # GPU list 
    import torch
    all_gpus = list(range(torch.cuda.device_count()))

    if len(all_gpus) == 0:
        raise RuntimeError("No CUDA device available.")

    print(f"Using GPUs: {all_gpus}")
    print(f"Datasets: {datasets}")

    ctx = mp.get_context('spawn')
    idx = 0
    while idx < len(datasets):
        procs = []
        for slot_idx, gpu_id in enumerate(all_gpus):
            if idx >= len(datasets):
                break
            ds = datasets[idx]
            port = args.port_base + slot_idx * args.port_stride
            p = ctx.Process(
                target=_worker,
                args=(ds, gpu_id, MODEL_PATH, args.file_name, port),
                daemon=False    
            )
            p.start()
            procs.append(p)
            idx += 1
        
        for p in procs:
            p.join()


if __name__ == "__main__":
    main()
