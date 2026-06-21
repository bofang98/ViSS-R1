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



TRANS_IMAGE_QUESTION = {
    "rotate": (
        "Please review the transformed image and determine which rotation angle was most likely applied to the original image. "
        "The possible options are:\n "
        "A. 0°\n B. 90°\n C. 180°\n D. 270°\n"
        "Note: The rotation is applied in the counterclockwise (anticlockwise) direction and 0° corresponds to no rotation.\n"
    ),
    "flip": (
        "Please review the input image and determine whether the image has been flipped. Note that the image could be flipped horizontally or vertically. "
        "The possible options are:\n "
        "A. No, the image remains in its original position.\n "
        "B. Yes, the image has been flipped \'horizontally\'.\n "
        "C. Yes, the image has been flipped \'vertically\'.\n "
    ),
    "puzzle": (
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
    )
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



def get_transform_question(example):
    media_type, trans_type, trans_ans = example['data_type'], example['trans_opt'], example['trans_ans']
    if media_type == 'image':
        question = TRANS_IMAGE_QUESTION[trans_type]
    elif media_type == 'video':
        question = TRANS_VIDEO_QUESTION[trans_type]
    return question



# pretext tasks
def process_transformation(image_inputs, video_inputs, example):
    trans_opt = example['trans_opt']        # rotate, flip, puzzle, arrow, shuffle 
    trans_ans = example['trans_ans']        # A, B, C, D, E, F

    if image_inputs: 
        image = image_inputs[0]
        
        if trans_opt == 'rotate':
            trans_image = process_rotate_image(image, trans_ans)
        elif trans_opt == 'flip':
            trans_image = process_flip_image(image, trans_ans)
        elif trans_opt == 'puzzle':
            trans_image = process_puzzle_image(image, trans_ans)
        return [trans_image], None,

    
    elif video_inputs:
        video = video_inputs[0]     # [16, 3, H, W]
        
        if trans_opt == 'rotate':
            trans_video = process_rotate_video(video, trans_ans)
        elif trans_opt == 'shuffle':
            trans_video = process_shuffle(video, trans_ans)
        elif trans_opt == 'arrow':
            trans_video = process_arrow(video, trans_ans)
        return None, [trans_video],



# rotate image
def process_rotate_image(image, trans_ans):
    """
    """
    # angles = [0, 90, 180, 270]
    # angle = random.choice(angles)
    gt_codebook = {
        0: "A",
        90: "B",
        180: "C",
        270: "D"
    }
    rev = {v: k for k, v in gt_codebook.items()}
    angle = rev.get(trans_ans)
    rotated_image = image.rotate(angle, expand=True)
    return rotated_image


def process_flip_image(image, trans_ans):
    """
    """
    gt_codebook = {
        'none': "A",
        'horizontal': "B",
        'vertical': "C",
    }
    rev = {v: k for k, v in gt_codebook.items()}
    angle = rev.get(trans_ans)
    if angle == 'horizontal':
        trans_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # gt_answer = "<transform>B</transform>"
    elif angle == 'vertical':
        trans_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        # gt_answer = "<transform>C</transform>"
    else:
        trans_image = image
        # gt_answer = "<transform>A</transform>"
    
    return trans_image


def process_puzzle_image(image, trans_ans):
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
    # idx1, idx2 = sorted(random.choice(all_pairs))
    answer_map = {
        (0, 1): 'A',
        (0, 2): 'B',
        (0, 3): 'C',
        (1, 2): 'D',
        (1, 3): 'E',
        (2, 3): 'F'
    }

    rev = {v: k for k, v in answer_map.items()}
    idx1, idx2 = rev.get(trans_ans)     # (0, 1)

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
    
    return new_img


# rotate transformation
def process_rotate_video(video, trans_ans):
    """
    """
    gt_codebook = {
        0: "A",
        90: "B",
        180: "C",
        270: "D"
    }
    rev = {v: k for k, v in gt_codebook.items()}
    angle = rev.get(trans_ans)  # 90

    # 对每一帧都应用同样的旋转
    rotated_frames = []
    for frame in video:  # frame: [3, H, W] tensor
        frame = frame / 255.0
        pil_img = TF.to_pil_image(frame)
        pil_rotated = pil_img.rotate(angle, expand=True)
        rotated_frame = TF.to_tensor(pil_rotated)
        rotated_frames.append(rotated_frame * 255)
    rotated_video = torch.stack(rotated_frames, dim=0)  # [16, 3, H, W] (或旋转后为[16, 3, W, H]，取决于角度)

    return rotated_video


def process_arrow(video, trans_ans):
    """
    """
    if trans_ans == 'B':
        trans_video = video.flip(0)
    elif trans_ans == 'A':
        trans_video = video
    
    return trans_video


def process_shuffle(video, trans_ans):
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
        (0, 1): 'A',
        (0, 2): 'B',
        (0, 3): 'C',
        (1, 2): 'D',
        (1, 3): 'E',
        (2, 3): 'F'
    }
    rev = {v: k for k, v in answer_map.items()}
    idx1, idx2 = rev.get(trans_ans)     # (0, 1)

    new_groups = groups.copy()
    new_groups[idx1], new_groups[idx2] = new_groups[idx2], new_groups[idx1]
    new_order = sum(new_groups, [])
    
    trans_video = video[new_order]

    return trans_video