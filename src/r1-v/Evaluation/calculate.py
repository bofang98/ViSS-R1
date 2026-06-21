import json
from collections import Counter, defaultdict

# 读取 JSON 文件
with open('eval_vsibench_VSIBench_clip_1_greedy_output.json', 'r', encoding='utf-8') as f:
    split_1 = json.load(f)

with open('eval_vsibench_VSIBench_clip_2_greedy_output.json', 'r', encoding='utf-8') as f:
    split_2 = json.load(f)

with open('eval_vsibench_VSIBench_clip_3_greedy_output.json', 'r', encoding='utf-8') as f:
    split_3 = json.load(f)

with open('eval_vsibench_VSIBench_clip_4_greedy_output.json', 'r', encoding='utf-8') as f:
    split_4 = json.load(f)

with open('eval_vsibench_VSIBench_clip_5_greedy_output.json', 'r', encoding='utf-8') as f:
    split_5 = json.load(f)


# 获取 "results" 字段
split_1 = split_1.get("results", [])
split1 = [item for item in split_1 if item.get('problem_type') == "multiple choice"]
split_2 = split_2.get("results", [])
split2 = [item for item in split_2 if item.get('problem_type') == "multiple choice"]
split_3 = split_3.get("results", [])
split3 = [item for item in split_3 if item.get('problem_type') == "multiple choice"]
split_4 = split_4.get("results", [])
split4 = [item for item in split_4 if item.get('problem_type') == "multiple choice"]
split_5 = split_5.get("results", [])
split5 = [item for item in split_5 if item.get('problem_type') == "multiple choice"]


accuracy1 = sum(item.get('reward', 0) for item in split1)/len(split1)*100
accuracy2 = sum(item.get('reward', 0) for item in split2)/len(split2)*100
accuracy3 = sum(item.get('reward', 0) for item in split3)/len(split3)*100
accuracy4 = sum(item.get('reward', 0) for item in split4)/len(split4)*100
accuracy5 = sum(item.get('reward', 0) for item in split5)/len(split5)*100
print(accuracy1, '\n', accuracy2, '\n', accuracy3, '\n', accuracy4, '\n', accuracy5)


all_filtered_results = [split1, split2, split3, split4, split5]

# 1. 收集所有problem_id
problem_ids = set()
for results in all_filtered_results:
    for item in results:
        problem_ids.add(item['problem_id'])

# 2. 构建 problem_id -> [item1, item2, ...] 的映射
problem_id_to_items = defaultdict(list)
for results in all_filtered_results:
    for item in results:
        problem_id_to_items[item['problem_id']].append(item)

# 3. 投票
final_results = []
for pid in problem_ids:
    items = problem_id_to_items[pid]
    # 收集所有prediction
    predictions = [item['prediction'] for item in items]
    # 投票，找出现次数最多的prediction
    vote_prediction = Counter(predictions).most_common(1)[0][0]
    for item in items:
        if item['prediction'] == vote_prediction:
            final_results.append(item.copy())
            break
import pdb; pdb.set_trace()
accuracy_vote = sum(item.get('reward', 0) for item in final_results)/len(final_results)*100
print(f"vote accuracy is {accuracy_vote}")