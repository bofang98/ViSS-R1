import torch
import json
import argparse
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils import *
from pprint import pprint



def get_argument_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--transformation", action='store_true')
    parser.add_argument("--split", type=int, default=1)
    args = parser.parse_args()
    return args


def init_model():
    # MODEL_PATH = "/gpubox11/disk2/wuwenhao/backup/ckpt/Qwen2.5-VL-7B-Instruct/"
    # MODEL_PATH = "/gpubox03/ssd5/fangbo05/Video-R1/Video-R1-7B"
    # MODEL_PATH = "/gpubox03/ssd5/fangbo05/VideoRFT-7B"
    # MODEL_PATH = "/gpubox03/ssd3/fangbo05/Qwen2.5-VL-7B-Stage-I-Pretex-Stage-II-Normal/checkpoint-1500/"
    MODEL_PATH = "/gpubox11/disk1/fangbo05/RL/ssl-cot-lr1e6-RL-lr1e6-Video-SSL-reward-0.5-wo-KL/checkpoint-2000"

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # default processer
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    # processor.tokenizer = tokenizer
    
    return model, processor, tokenizer




def update_ssl_messgae(example, trans_gt, trans_prompt):
    """Prepare dataset example for CoT generation."""

    if example["problem_type"] == 'multiple choice':
        question = example['problem'] + "Options:\n"
        for op in example["options"]:
            question += op + "\n"
    else:
        question = example['problem']
    

    system_message = (
        "You are an advanced multimedia analysis assistant.\n"
        # "All your thinking processes should be a concise, logical, visual-based chain-of-thought, less than 300 tokens. "
        # "Think naturally as a human would, using inner dialogue like 'let me think', 'hmm', 'wait', etc. "
        # "All your formal output should be a brief sentence, less than 50 tokens."
    )
    


    SSL_THINKING_TEMPLATE = (
        "You will be given an image or video that has been transformed in some way, along with two progressive questions:\n"
        "1. Transformation Identification: Based on the visual content, identify the type of transformation applied to the image or video.\n"
        "2. User Question: On the basis of the restored (inverse-transformed) image or video, answer the real user's question. You need to think even more deeply about the user question.\n"
        "Please think through both questions step by step, as a human would. Carefully analyze the visual content and reason in detail. "
        "Your thought process should include natural inner dialogue such as 'let me think,' 'hmm,' 'wait,' 'oh, I see,' 'let’s break it down,' etc., and show self-reflection or verification as you work through each problem. "
        "Put your detailed thought process inside <think>...</think> tags. Place your final transformation identification inside <transform>...</transform> tags. Place your final answer to the user question inside <answer>...</answer> tags. For example: <think> ... </think><transform> ... </transform><answer> ... </answer>.\n"
        "Important:\n"
        "- Your reasoning and answer must be based on the visual information from the input image or video.\n"
        "- Do not mention or refer to any 'ground truth', 'reference answer' or similar terms in your output.\n"
        "- Your transformation identification must exactly match the provided Ground Truth Transform, and your answer to the user question must exactly match the provided Ground Truth Answer.\n"
        "Transform question: {Trans_question}\n"
        "Ground Truth Transform: {Trans_answer}\n"
        "User Question: {Question}\n"
        "Ground Truth Answer: {Answer}"
    )

    

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": example['data_type'],
                    # example['data_type']: os.getcwd() + "/Video-R1-data" + example['path'][1:]
                    example['data_type']: "/gpubox03/ssd3/fangbo05/Video-R1-data" + example['path'][1:],
                },
                {
                    "type": "text",
                    # "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[x['problem_type']],
                    "text": SSL_THINKING_TEMPLATE.format(Trans_question=trans_prompt, Trans_answer=trans_gt, Question=question, Answer=example['solution']),
                }
            ]
        }
    ]
    

    return messages




def prepare_messgae(example):
    """Prepare dataset example for CoT generation."""

    system_message = (
        "You are an advanced multimedia analysis assistant.\n"
        # "All your thinking processes should be a concise, logical, visual-based chain-of-thought, less than 300 tokens. "
        # "Think naturally as a human would, using inner dialogue like 'let me think', 'hmm', 'wait', etc. "
        # "All your formal output should be a brief sentence, less than 50 tokens."
    )
    
    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    THINKING_TEMPLATE = (
        "Please think about this question as if you were a human pondering deeply. "
        "When answering the user's question, please carefully analyze and reason step by step, making sure to incorporate and reference the visual content of the input image or video as part of your thought process. "
        "Think as a human would: use natural internal dialogue such as 'let me think', 'hmm', 'wait', 'oh, I see', 'let's break it down', etc., and include self-reflection or verification as you reason through the problem. "
        "Your thought process should be output within <think>...</think> tags, and your final answer within <answer>...</answer> tags. For example: <think> ... </think><answer> ... </answer>.\n"
        "Important:\n"
        "- Your reasoning and answer must be based on the visual information from the input image or video.\n"
        "- Do not mention or refer to any 'ground truth', 'reference answer' or similar terms in your output.\n"
        "- You must arrive at your final answer solely through step-by-step visual reasoning, and your final answer must match the provided Ground Truth Answer exactly.\n"
        "User Question: {Question}\n"
        "Ground Truth Answer: {Answer}"
    )


    if example["problem_type"] == 'multiple choice':
        question = example['problem'] + "Options:\n"
        for op in example["options"]:
            question += op + "\n"
    else:
        question = example['problem']


    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": example['data_type'],
                    # example['data_type']: os.getcwd() + "/Video-R1-data" + example['path'][1:]
                    example['data_type']: "/gpubox03/ssd3/fangbo05/Video-R1-data" + example['path'][1:],
                    # "max_pixels": 360*420,
                    # "fps": 1.0
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']],
                    # "text": THINKING_TEMPLATE.format(Question=question, Answer=example['solution']),
                }
            ]
        }
    ]
    

    return messages



def main(args):
    # init model
    model, processor, tokenizer = init_model()

    # source file with videos and solutions
    dataset_name = "Video-R1-Video-260k"
    PROMPT_PATH = f"./{dataset_name}.json"
        
    data = []
    if PROMPT_PATH.endswith('.jsonl'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    elif PROMPT_PATH.endswith('.json'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Input file must be .json or .jsonl")

    # split data into 20 blocks: 165575 // 20
    len_block = len(data) // 20     # 8278
    start_index = args.split * len_block
    end_index = (args.split + 1) * len_block
    split_data = data[start_index: end_index]
    print(f"Split number: {args.split}---------------------------------------------------------")
    print('Reason data from Start index: {} to End index: {}'.format(start_index, end_index))

    OUTPUT_PATH = f"./outputs/{dataset_name}-{args.split}.json"



    final_output = []

    for example in tqdm(split_data):

        message = prepare_messgae(example)
        
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
        
        if args.transformation:
            # apply ssl transformation, switch the initial image and video
            image_inputs, video_inputs, trans_gt, trans_prompt, trans_opt = process_transformation(image_inputs, video_inputs)
            # replace with new message and text
            message = update_ssl_messgae(example, trans_gt, trans_prompt)
            text = processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            print(f"SSL Transformation: {message}")

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")
        

        # visual_model = model.visual
        # visual_model.to("cuda")

        # lm_model = model.model
        # lm_model.to("cuda")

        # textual embedding
        # inputs_ids = inputs["input_ids"].type(torch.long)
        # text_embeds = lm_model(inputs_ids).last_hidden_state
        # print(text_embeds.shape)
 
        with torch.no_grad():
            # visual embedding
            if image_inputs:
                pixel_values = inputs["pixel_values"].type(torch.bfloat16)
                image_embeds = model.visual(pixel_values, grid_thw=inputs["image_grid_thw"])
            elif video_inputs:
                pixel_values = inputs["pixel_values_videos"].type(torch.bfloat16)
                video_embeds = model.visual(pixel_values, grid_thw=inputs["video_grid_thw"])        # [xxx, 3584]
            print("video embedding shape: ", video_embeds.shape, video_embeds.dtype)

        
        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        answer_id = generated_ids_trimmed[0].unsqueeze(0).type(torch.long)
        answer_embed = model.model(answer_id)[0].squeeze()      # [xxx, 3584]
        print("answer embedding: ", answer_embed.shape, answer_embed.dtype)

        prompt_q = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to("cuda")
        prompt_embed = model.model(prompt_q)[0].squeeze()
        print("prompt embedding: ", prompt_embed.shape)

        # save features
        torch.save(video_embeds, f"/root/paddlejob/workspace/env_run/log/cache/VideoSSR/video_embed_{example['problem_id']}.pt")
        torch.save(answer_embed, f"/root/paddlejob/workspace/env_run/log/cache/VideoSSR/answer_embed_{example['problem_id']}.pt")
        torch.save(prompt_embed, f"/root/paddlejob/workspace/env_run/log/cache/VideoSSR/prompt_embed_{example['problem_id']}.pt")
        import pdb; pdb.set_trace()



if __name__ == '__main__':
    args = get_argument_parser()
    main(args)




"""
script
"""

# source /root/paddlejob/workspace/env_run/ENV/env_vision_video_r1_h100.sh
# CUDA_VISIBLE_DEVICES=7 python anna_7b_model.py --split 0 
# CUDA_VISIBLE_DEVICES=1 python anna_7b_model.py --split 0