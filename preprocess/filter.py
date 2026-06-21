import os
import json
import torch
import argparse
from tqdm import tqdm


def main():
    with open("./Video-R1-Video-260k-filter.json", 'r') as f:
        all_data = json.load(f)     
    print(f"Total data: {len(all_data)}")


    for item in tqdm(all_data):
        loss_file = os.path.join('./output_mask_loss', str(item["problem_id"])+'.json')
        with open(loss_file, 'r') as f:
            data = json.load(f)
            num_frames = len(json.loads(data[0]['mask']))  
            
            loss_sum = torch.zeros(num_frames)
            count = torch.zeros(num_frames)

            for item_mask in data:
                mask = torch.tensor(json.loads(item_mask['mask']), dtype=torch.float32) 
                loss = float(item_mask['loss'])
                
                loss_sum += mask * loss
                count += mask

            loss_distribution = loss_sum / count
        
        item["loss"] = torch.mean(loss_distribution).item()



    # ranking
    sorted_data = sorted(all_data, key=lambda x:x["loss"], reverse=True)

    n = len(sorted_data)
    # start_index = int(n * 0.2)
    # end_index = int(n * 0.8)

    end_index = int(n*0.5)

    filtered_data = sorted_data[end_index:]
    
    # save 
    with open("./Video-R1-Video-260k-low-loss.json", 'w') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)



def filter_value_0(input_json="./Video-R1-260k.json", value_file="./value_data_0.txt", output_json="./Video-R1-260k-value-loss.json"):
    with open(input_json, 'r') as f:
        all_data = json.load(f)     
    print(f"Total data: {len(all_data)}")
    
    problem_ids = []

    with open(value_file, 'r') as f:
        for line in f:
            line = line.strip()
            data = eval(line)
            problem_ids.append(data['problem_id'])
    
    filter_value_loss = []
    for i, item in enumerate(tqdm(all_data)):
        if item['problem_id'] in problem_ids:
            filter_value_loss.append(item)

    with open(output_json, 'w') as f:
        json.dump(filter_value_loss, f, indent=4, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", default="./Video-R1-260k.json")
    parser.add_argument("--value_file", default="./value_data.txt")
    parser.add_argument("--output_json", default="./Video-R1-260k-value-video.json")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.input_json, 'r') as f:
        all_data = json.load(f)     
    print(f"Total data: {len(all_data)}")
    
    problem_ids = []

    with open(args.value_file, 'r') as f:
        for line in f:
            line = line.strip()
            data = eval(line)
            problem_ids.append(data['problem_id'])


    filter_value_loss = []
    for i, item in enumerate(tqdm(all_data)):
        if item['problem_id'] in problem_ids:
            filter_value_loss.append(item)

    with open(args.output_json, 'w') as f:
        json.dump(filter_value_loss, f, indent=4, ensure_ascii=False)
