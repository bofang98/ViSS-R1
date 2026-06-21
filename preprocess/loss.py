import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

torch.manual_seed(42)

vl_model_path = os.environ.get("VL_MODEL_PATH", "/path/to/Qwen2.5-VL-7B-Instruct")

processor = AutoProcessor.from_pretrained(vl_model_path)
tokenizer = AutoTokenizer.from_pretrained(vl_model_path)


gt_answer = "A women and a dog sit on a beach. The women smiles and looks softly towards her pet dog."
# gt_answer = "The image shows a strong tiger lying on the grass."
concat_text = "Describe this simage." + " " + gt_answer

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "file://./tiger.jpeg"},
        {"type": "text", "text": concat_text}
    ],
}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

selected_frames = torch.randint(low=0, high=256, size=(1,3,112,224), dtype=torch.int)

inputs = processor(text=[text], images=None, videos=[selected_frames], padding=True,return_tensors="pt",padding_side='left')
import pdb; pdb.set_trace()

### 
messages_del = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "file://./tiger.jpeg"},
        {"type": "text", "text": "Describe this simage."}
    ],
}]
text_del = processor.apply_chat_template(messages_del, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages_del)

inputs_del = processor(
    text=[text_del],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)
question_len_del = inputs_del['input_ids'].shape[1]

###

model_flash = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    vl_model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": 0},
)
model_flash.eval()


# 构造labels，只在答案部分有值，其余为-100
labels = inputs["input_ids"].clone().to(model_flash.device)
question_len = question_len_del - 1  # 减1因为BOS
labels[0, :question_len] = -100  # 忽略问题部分

with torch.no_grad():
    inputs_flash = {k: v.to("cuda:0") for k, v in inputs.items()}
    # out_flash = model_flash(**inputs).logits.cpu()
    out_flash = model_flash(input_ids=inputs_flash["input_ids"], attention_mask=inputs_flash["attention_mask"], labels=labels)
    print(gt_answer, labels.shape)
    print(out_flash)
    # out_flash = model_flash(input_ids=inputs_flash["input_ids"], attention_mask=inputs_flash["attention_mask"], labels=gt_answer)




# beach <> gt1  --> loss: 9.1270
# tiger <> gt1  --> loss: 9.6706
# tiger <> gt2  --> loss: 6.9134
# beach <> gt2  --> loss: 9.3735
