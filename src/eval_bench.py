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
import torchvision.transforms.functional as TF
import random



def set_seed(seed=42):
    # Python内置随机数
    random.seed(seed)
    # Numpy随机数
    np.random.seed(seed)
    # Python哈希种子（影响部分操作）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 如果用PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass





def process_video(video_inputs):
    video = video_inputs    # [32, 3, H, W]
    trans_opt = random.choice(['rotate', 'arrow', 'shuffle'])
    if trans_opt == 'rotate':
        trans_video, trans_gt, trans_question = process_rotate_video(video)
    elif trans_opt == 'shuffle':
        trans_video, trans_gt, trans_question = process_shuffle(video)
    elif trans_opt == 'arrow':
        trans_video, trans_gt, trans_question = process_arrow(video)
    return trans_video, trans_gt, trans_question




# rotate transformation
def process_rotate_video(video):
    """
    """
    angles = [0, 90, 180, 270]
    angle = random.choice(angles)
    gt_codebook = {
        0: "<transform>A</transform>",
        90: "<transform>B</transform>",
        180: "<transform>C</transform>",
        270: "<transform>D</transform>"
    }
    gt_answer = gt_codebook[angle]
    # 对每一帧都应用同样的旋转
    rotated_frames = []
    for frame in video:  # frame: [3, H, W] tensor
        frame = frame / 255.0
        pil_img = TF.to_pil_image(frame)
        pil_rotated = pil_img.rotate(angle, expand=True)
        rotated_frame = TF.to_tensor(pil_rotated)
        rotated_frames.append(rotated_frame * 255)
    rotated_video = torch.stack(rotated_frames, dim=0)  # [16, 3, H, W] (或旋转后为[16, 3, W, H]，取决于角度)

    trans_question = (
        "Please review the transformed video clip and determine which rotation angle was most likely applied to the original video. "
        "The possible options are:\n "
        "A. 0°\n B. 90°\n C. 180°\n D. 270°\n"
        "Note: The rotation is applied in the counterclockwise (anticlockwise) direction and 0° corresponds to no rotation.\n"
    )
    return rotated_video, gt_answer, trans_question


def process_arrow(video):
    """
    """
    if random.random() < 0.55:
        trans_video = video.flip(0)
        gt_answer = "<transform>B</transform>"
    else:
        trans_video = video
        gt_answer = "<transform>A</transform>"
    trans_question = (
        "Please review the input video clip and determine whether the sequence of video frames has been reversed (i.e., played in reverse order). "
        "The possible options are:\n "
        "A. No, the video frames remain in their original order.\n "
        "B. Yes, the video frames have been reversed.\n"
    )
    return trans_video, gt_answer, trans_question


def process_shuffle(video):
    """
    select two groups, and switch their position
    """
    assert video.shape[0] == 32, "video lenght is not 32 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    groups = [
        list(range(0, 8)),   # [0,1,2,3]
        list(range(8, 16)),   # [4,5,6,7]
        list(range(16, 24)),  # [8,9,10,11]
        list(range(24, 32))  # [12,13,14,15]
    ]
    answer_map = {
        (0, 1): '<transform>A</transform>',
        (0, 2): '<transform>B</transform>',
        (0, 3): '<transform>C</transform>',
        (1, 2): '<transform>D</transform>',
        (1, 3): '<transform>E</transform>',
        (2, 3): '<transform>F</transform>'
    }

    idx1, idx2 = sorted(random.sample(range(4), 2))
    new_groups = groups.copy()
    new_groups[idx1], new_groups[idx2] = new_groups[idx2], new_groups[idx1]
    new_order = sum(new_groups, [])
    
    trans_video = video[new_order]
    gt_answer = answer_map[(idx1, idx2)]
    trans_question = (
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

    return trans_video, gt_answer, trans_question










set_seed(42)
BSZ = 64


parser = argparse.ArgumentParser(description="Evaluation benchmark")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
parser.add_argument('--file_name', type=str, required=True, help="Name of the file")
parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID")
args = parser.parse_args()

MODEL_PATH = args.model_path
file_name = args.file_name



llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len = 8192 * 2,
    gpu_memory_utilization=0.8,
    limit_mm_per_prompt={"image": 1, "video": 1},
    task="embed",       # default is "generate"
    enforce_eager=True,
)



sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    max_tokens=1024,
    stop_token_ids=[],
)


processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer


# for dataset_name in ['mmvu','vsibench', 'videommmu','mvbench','videomme','tempcompass']:     # 'mmvu','vsibench','videommmu','mvbench','videomme','tempcompass'
for dataset_name in ['videomme']:

    OUTPUT_PATH = f"./src/r1-v/eval_results_tr/eval_{dataset_name}_{file_name}_greedy_output.json"
    PROMPT_PATH = f"./src/r1-v/Evaluation/eval_{dataset_name}.json"
    
    if PROMPT_PATH.endswith('.jsonl'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    elif PROMPT_PATH.endswith('.json'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Input file must be .json or .jsonl")

    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
    )

    # QUESTION_TEMPLATE = (
    #     "You will be given a video that has been transformed in some way, along with two progressive questions:\n"
    #     "1. Transformation Identification: Based on the visual content, identify the type of transformation applied to the image or video.\n"
    #     "2. User Question: On the basis of the restored (inverse-transformed) image or video, answer the real user's question. You need to think even more deeply about the user question.\n"
    #     "Please think through both questions step by step, as a human would. Carefully analyze the visual content and reason in detail. "
    #     "Your thought process should include natural inner dialogue such as 'let me think,' 'hmm,' 'wait,' 'oh, I see,' 'let’s break it down,' etc., and show self-reflection or verification as you work through each problem. "
    #     "Put your detailed thought process inside <think>...</think> tags. Place your final transformation identification inside <transform>...</transform> tags. Place your final answer to the user question inside <answer>...</answer> tags. For example: <think> ... </think><transform> ... </transform><answer> ... </answer>.\n"
    #     "Important:\n"
    #     "- Your reasoning and answer must be based on the visual information from the input image or video.\n"
    #     "- For answering the Transform Question, provide only the single option letter (e.g., A, B, etc.) within the <transform> </transform> tags.\n"
    #     "- For answering the User Question, {Type_template}\n"
    #     "Transform question: {Trans_question}\n"
    #     "User Question: {Question}"
    # )


    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }


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
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[x['problem_type']]
                    # "text": QUESTION_TEMPLATE.format(Type_template=TYPE_TEMPLATE[x['problem_type']], Question=question, Trans_question="")
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
                print(f"Resuming from sample index {start_idx}")
        except Exception as e:
            print(f"Error reading existing output file: {e}")

    
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
    for i in tqdm(range(start_idx, len(messages), BSZ), desc="Processing batches"):
        batch_messages = messages[i:i + BSZ]

        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        

        image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)    # [64x [32, 3, H, W]]

        # apply random video transformations
        # new_video_inputs = []
        # for xx_idx in range(len(video_inputs)):
        #     trans_video, trans_gt, trans_question = process_video(video_inputs[xx_idx])
        #     prompts[xx_idx] = prompts[xx_idx].replace('Transform question: \n', f"Transform question: {trans_question}\n")
        #     new_video_inputs.append(trans_video)
        # video_inputs = new_video_inputs

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

        outputs = llm.embed(llm_inputs)       # 3025
        # outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
        # batch_output_text = [out.outputs[0].text for out in outputs]

        # save embedding
        group_1_embed = []
        for xx in outputs:
            embed = torch.tensor(xx.outputs.embedding)
            group_1_embed.append(embed)
        group_1_embed = torch.stack(group_1_embed, dim=0)
        print(group_1_embed.shape, 'xx'*30)
        torch.save(group_1_embed, f"/root/paddlejob/workspace/env_run/log/cache/VideoSSR/group_1_ssr1.pt")
        import pdb; pdb.set_trace()    
            
        # except Exception as e:
        #     print('error:', data[i]['path'])
        #     print('Exception:', e)
        #     batch_output_text = ['<answer>error</answer>'] * BSZ
            

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
            print(f"Processed batch {(i - start_idx)//BSZ + 1}, saved {len(final_output)} samples.")
        except Exception as e:
            print(f"Error writing to output file: {e}")

    final_acc={'mean_acc': 0.0, 'mean_mra': 0.0}
    final_acc['mean_acc'] = torch.tensor(mean_acc).mean().item()
    if mean_mra != []:
        final_acc['mean_mra'] = torch.tensor(mean_mra).mean().item()
    
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": final_output, "final_acc": [final_acc]}, f, indent=2, ensure_ascii=False)
        print(f"Final accuracy saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error writing final accuracy to output file: {e}")
    
    print(f"Results saved to {OUTPUT_PATH}")

