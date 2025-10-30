import json

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_user_ids_from_prompts(prompts):
    user_ids = set()
    for entry in prompts:
        user_id = entry.get('user_id')
        if user_id:
            user_ids.add(user_id)
    return user_ids

def main():
    file1 = 'sft/generated_prompts.json'
    file2 = 'checkpoint_test_result/rrec_622_checkpoint_6058_do_sample.json'

    # 读取第一个文件（prompts）
    data1 = load_json(file1)
    prompts1 = data1['prompts'] if isinstance(data1, dict) and 'prompts' in data1 else data1
    user_ids1 = get_user_ids_from_prompts(prompts1)

    # 读取第二个文件（detailed_results）
    data2 = load_json(file2)
    prompts2 = data2['detailed_results'] if isinstance(data2, dict) and 'detailed_results' in data2 else data2
    user_ids2 = get_user_ids_from_prompts(prompts2)

    # 查找交集
    common_user_ids = user_ids1 & user_ids2
    if common_user_ids:
        print(f"存在相同的user_id: {common_user_ids}")
    else:
        print("没有相同的user_id")

if __name__ == '__main__':
    main()
