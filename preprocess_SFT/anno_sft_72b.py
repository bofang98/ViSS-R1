import os
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


def resolve_media_path(example):
    data_root = os.environ.get("VIDEO_R1_DATA_ROOT", "./Video-R1-data")
    return os.path.join(data_root, example["path"].lstrip("/"))


def init_model():
    MODEL_PATH = os.environ.get("QWEN72B_MODEL_PATH", "/path/to/Qwen2.5-VL-72B-Instruct")

    # default: Load the model on the available device(s)
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     MODEL_PATH, torch_dtype="auto", device_map="auto"
    # )

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
                    example['data_type']: resolve_media_path(example),
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
                    example['data_type']: resolve_media_path(example),
                },
                {
                    "type": "text",
                    # "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[x['problem_type']],
                    "text": THINKING_TEMPLATE.format(Question=question, Answer=example['solution']),
                }
            ]
        }
    ]
    

    return messages



def main(args):
    # init model
    model, processor, tokenizer = init_model()

    # source file with videos and solutions
    dataset_name = "Video-R1-COT-165k"
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
        
        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)

        # extract
        think_chain = extract_think(output_text[0])
        final_ans = extract_answer(output_text[0])
        if args.transformation: 
            trans_ans = extract_transform(output_text[0])

        # example["answer"] = final_ans
        q_type = example.get("problem_type", "")
        # example["reward"] = reward_fn(example, output_text[0], q_type)
        # example['select'] = True if example["reward"] > 0.8 else False
        if think_chain:
            example["process"] = f"<think>{think_chain}</think>"        # replace original think chain
        example['trans_opt'] = trans_opt
        example['trans_ans'] = trans_ans
        final_output.append(example)

        # save
        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing to output file: {e}")



if __name__ == '__main__':
    args = get_argument_parser()
    main(args)




"""
script
"""

# Example:
# CUDA_VISIBLE_DEVICES=0,1 python anno_sft_72b.py --split 17 --transformation 
# CUDA_VISIBLE_DEVICES=2,3 python anno_sft_72b.py --split 18 --transformation 
# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
# 14 40G
