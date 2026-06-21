import os
import json
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
import random

# from transformers import AutoProcessor, AutoTokenizer
# from vllm import LLM, SamplingParams
# from qwen_vl_utils import process_vision_info

import torchvision.transforms.functional as TF
from PIL import Image

import copy


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

def extract_transform(text):
    pattern = r'<transform>\s*(.*?)\s*</transform>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except Exception as e:
        print(f"Error converting '{num_str}' to float: {e}")
        return None

def wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    m = len(ref_words)
    n = len(hyp_words)
    d = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        d[i][0] = i
    for j in range(n+1):
        d[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[m][n] / max(1, m)

def compute_bleu_score(reference, hypothesis):
    try:
        smoothing = SmoothingFunction().method1
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
        return score
    except Exception as e:
        print(f"Error computing BLEU score: {e}")
        return 0.0

def compute_rouge_score(reference, hypothesis, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
    scores = scorer.score(reference, hypothesis)
    average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
    return average_fmeasure

def reward_fn(sample, model_output, question_type):
    try:
        output_ans = extract_answer(model_output)
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
        elif question_type == "OCR":
            error_rate = wer(gt_ans, output_ans)
            reward = 1 - error_rate
            return max(0.0, min(1.0, reward))
        elif question_type == "free-form":
            score = compute_rouge_score(gt_ans, output_ans)
            return max(0.0, min(1.0, score))
        elif question_type == "regression":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
            rel_diff = min(1.0, max(0.0, rel_diff))
            return 1 - rel_diff
        else:
            return 0.0
    except Exception as e:
        print(f"Error in reward_fn for question_type '{question_type}': {e}")
        return 0.0





# pretext tasks
def process_transformation(image_inputs, video_inputs):
    if image_inputs: 
        image = image_inputs[0]
        trans_opt = random.choice(['rotate', 'flip', 'puzzle'])
        if trans_opt == 'rotate':
            trans_image, trans_gt, trans_prompt = process_rotate_image(image)
        elif trans_opt == 'flip':
            trans_image, trans_gt, trans_prompt = process_flip_image(image)
        elif trans_opt == 'puzzle':
            trans_image, trans_gt, trans_prompt = process_puzzle_image(image)
        return [trans_image], None, trans_gt, trans_prompt, trans_opt

    
    elif video_inputs:
        video = video_inputs[0]     # [16, 3, H, W]
        trans_opt = random.choice(['rotate', 'arrow', 'shuffle'])
        if trans_opt == 'rotate':
            trans_video, trans_gt, trans_prompt = process_rotate_video(video)
        elif trans_opt == 'shuffle':
            trans_video, trans_gt, trans_prompt = process_shuffle(video)
        elif trans_opt == 'arrow':
            trans_video, trans_gt, trans_prompt = process_arrow(video)
        return None, [trans_video], trans_gt, trans_prompt, trans_opt



# rotate image
def process_rotate_image(image):
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
    rotated_image = image.rotate(angle, expand=True)
    trans_prompt = (
        "Please review the transformed image and determine which rotation angle was most likely applied to the original image. "
        "The possible options are:\n "
        "A. 0°\n B. 90°\n C. 180°\n D. 270°\n"
        "Note: The rotation is applied in the counterclockwise (anticlockwise) direction and 0° corresponds to no rotation.\n"
        # "Please foucs on the spatial cues of the image and think about this question as if you were a human pondering deeply. "
        # "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. "
        # "It's encouraged to include self-reflection or verification in the reasoning process. "
        # "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags. "
        # "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
        )
    return rotated_image, gt_answer, trans_prompt


def process_flip_image(image):
    """
    """
    angle = random.choice(['horizontal', 'vertical', 'none'])
    if angle == 'horizontal':
        trans_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        gt_answer = "<transform>B</transform>"
    elif angle == 'vertical':
        trans_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        gt_answer = "<transform>C</transform>"
    else:
        trans_image = image
        gt_answer = "<transform>A</transform>"
    trans_prompt = (
        "Please review the input image and determine whether the image has been flipped. Note that the image could be flipped horizontally or vertically. "
        "The possible options are:\n "
        "A. No, the image remains in its original position.\n "
        "B. Yes, the image has been flipped \'horizontally\'.\n "
        "C. Yes, the image has been flipped \'vertically\'.\n "
        # "Please focus on the image contents and think about this question as if you were a human pondering deeply. "
        # "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. "
        # "It's encouraged to include self-reflection or verification in the reasoning process. "
        # "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags. "
        # "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
    )
    return trans_image, gt_answer, trans_prompt


def process_puzzle_image(image):
    """"""
    w, h = image.size   
    patch_w, patch_h = w // 2, h // 2   # 4 patches
    patches = []
    for i in range(2):
        for j in range(2):
            left = j * patch_w
            upper = i * patch_h
            right = left + patch_w
            lower = upper + patch_h
            patch = image.crop((left, upper, right, lower))
            patches.append(patch)
    
    all_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    idx1, idx2 = sorted(random.choice(all_pairs))
    answer_map = {
        (0, 1): '<transform>A</transform>',
        (0, 2): '<transform>B</transform>',
        (0, 3): '<transform>C</transform>',
        (1, 2): '<transform>D</transform>',
        (1, 3): '<transform>E</transform>',
        (2, 3): '<transform>F</transform>'
    }
    gt_answer = answer_map[(idx1, idx2)]

    # swap two patches
    new_patches = patches.copy()
    new_patches[idx1], new_patches[idx2] = new_patches[idx2], new_patches[idx1]
    
    # reconstruct a new image
    new_img = Image.new('RGB', (patch_w*2, patch_h*2))
    idx = 0
    for i in range(2):
        for j in range(2):
            new_img.paste(new_patches[idx], (j*patch_w, i*patch_h))
            idx += 1
    
    trans_prompt = (
        "Please carefullt review the transformed image. The image has been evenly divided into four equal-sized patches, labeled as follows:\n "
        "Patch 1: Top-Left\n Patch 2: Top-Right\n Patch 3: Bottom-Left\n Patch 4: Bottom-Right\n"
        "Two of these patches have been randomly selected and swapped with each other, while the other two patches remain in their original positions. "
        "Your task is to determine which pair of patches were swapped. "
        "Options:\n "
        "A. Patch 1 and Patch 2 were swapped\n "
        "B. Patch 1 and Patch 3 were swapped\n "
        "C. Patch 1 and Patch 4 were swapped\n "
        "D. Patch 2 and Patch 3 were swapped\n "
        "E. Patch 2 and Patch 4 were swapped\n "
        "F. Patch 3 and Patch 4 were swapped\n "
        # "Please focus on the spatial details and cues within the image contents and think about this question as if you were a human pondering deeply. "
        # "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. "
        # "It's encouraged to include self-reflection or verification in the reasoning process. "
        # "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags. "
        # "Please provide only the single option letter (e.g., A, B, C, D, E, etc.) within the <answer> </answer> tags."
    )
    return new_img, gt_answer, trans_prompt


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

    trans_prompt = (
        "Please review the transformed video and determine which rotation angle was most likely applied to the original video. "
        "The possible options are:\n "
        "A. 0°\n B. 90°\n C. 180°\n D. 270°\n"
        "Note: The rotation is applied in the counterclockwise (anticlockwise) direction and 0° corresponds to no rotation.\n"
        # "Please foucs on the spatial cues of video contents and think about this question as if you were a human pondering deeply. "
        # "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. "
        # "It's encouraged to include self-reflection or verification in the reasoning process. "
        # "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags. "
        # "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
        )
    return rotated_video, gt_answer, trans_prompt


def process_arrow(video):
    """
    """
    if random.random() < 0.55:
        trans_video = video.flip(0)
        gt_answer = "<transform>B</transform>"
    else:
        trans_video = video
        gt_answer = "<transform>A</transform>"
    trans_prompt = (
        "Please review the input video and determine whether the sequence of video frames has been reversed (i.e., played in reverse order). "
        "The possible options are:\n "
        "A. No, the video frames remain in their original order.\n "
        "B. Yes, the video frames have been reversed.\n"
        # "Please focus on the temporal order of video contents and think about this question as if you were a human pondering deeply. "
        # "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. "
        # "It's encouraged to include self-reflection or verification in the reasoning process. "
        # "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags. "
        # "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
    )
    return trans_video, gt_answer, trans_prompt


def process_shuffle(video):
    """
    select two groups, and switch their position
    """
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
    trans_prompt = (
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
        # "Please focus on the temporal continuity of video contents and think about this question as if you were a human pondering deeply. "
        # "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. "
        # "It's encouraged to include self-reflection or verification in the reasoning process. "
        # "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags. "
        # "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
    )

    return trans_video, gt_answer, trans_prompt




from pathlib import Path
def collect_save_results(folder="outputs"):
    merged = []
    for file_path in sorted(Path(folder).glob("*.json")):
        with file_path.open("r", encoding="utf-8") as fh:
            try:
                payload = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path} 无法解析：{exc}") from exc
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ValueError(f"{file_path} 中的 result 不是列表")
        merged.extend(results)
    
    print(f"合并了 {len(merged)} 个结果")

    # filter out results that having ground truth in their thinking process
    for result in tqdm(merged[:]):
        if "ground truth" in result['process']:
            merged.remove(result)
            # print(f"{result['id']} 被过滤掉")
        elif result['trans_ans'] == "":
            merged.remove(result)
        elif len(result['trans_ans']) > 1:
            # print(f"{result['trans_ans']}, see")
            # result['trans_ans'] = result['trans_ans'][0]
            merged.remove(result)

    print(f"过滤包含ground truth后剩余 {len(merged)} 个结果")
    # final check
    for item in tqdm(merged[:]):
        if item['trans_opt'] == 'rotate':
            assert item['trans_ans'] in ['A', 'B', 'C', 'D']
        elif item['trans_opt'] == 'shuffle' or item['trans_opt'] == 'puzzle':
            assert item['trans_ans'] in ['A', 'B', 'C', 'D', 'E', 'F']
        elif item['trans_opt'] == 'arrow':
            assert item['trans_ans'] in ['A', 'B']
        elif item['trans_opt'] == 'flip':
            assert item['trans_ans'] in ['A', 'B', 'C']

    # save
    out_path = Path(f"./SSL-COT-{len(merged)//1000}k.json")
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=4, ensure_ascii=False)
    return merged



if __name__ == "__main__":
    all_results = collect_save_results("./outputs_32frame")
    import pdb; pdb.set_trace()
    print("")