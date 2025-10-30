#!/usr/bin/env python3
"""
过滤出正确回答的样本并转换为训练格式
"""

import json
import re
import os


def extract_answer(model_response):
    """提取<answer>标签内的答案并标准化"""
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, model_response, re.IGNORECASE | re.DOTALL)
    
    if match:
        answer = match.group(1)
        # 去除标点符号和空格，转换为小写
        answer = re.sub(r'[^\w]', '', answer).lower()
        # 检查是否为有效答案
        if answer in ['yes', 'no']:
            return answer
        else:
            return None  # 提取到了内容但不是有效答案
    return None  # 没有找到answer标签


def is_correct_response(is_match, answer):
    """判断回答是否正确"""
    if answer is None:  # 提取失败
        return False
    if is_match and answer == "yes":
        return True
    elif not is_match and answer == "no":
        return True
    return False


def filter_correct_responses(input_file, output_file):
    """过滤正确的回答并转换为训练格式"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取文件 {input_file}: {e}")
        return
    
    if 'responses' not in data:
        print(f"错误: 文件中没有 'responses' 字段")
        return
    
    responses = data['responses']
    filtered_data = []
    
    correct_count = 0
    total_count = len(responses)
    extraction_failed_count = 0
    fail_no_answer_tag = 0
    fail_not_yes_no = 0
    fail_missing_field = 0
    fail_empty_response = 0
    fail_samples = []  # 存储部分失败样本
    
    for idx, response in enumerate(responses):
        # 检查必要字段是否存在
        if not all(key in response for key in ['is_match', 'model_response', 'input_prompt']):
            fail_missing_field += 1
            extraction_failed_count += 1
            if len(fail_samples) < 10:
                fail_samples.append({
                    'reason': 'missing_field',
                    'idx': idx,
                    'response': response
                })
            continue

        is_match = response['is_match']
        model_response = response['model_response']
        input_prompt = response['input_prompt']

        if not model_response or not isinstance(model_response, str):
            fail_empty_response += 1
            extraction_failed_count += 1
            if len(fail_samples) < 10:
                fail_samples.append({
                    'reason': 'empty_model_response',
                    'idx': idx,
                    'response': response
                })
            continue

        # 提取答案
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, model_response, re.IGNORECASE | re.DOTALL)
        if not match:
            fail_no_answer_tag += 1
            extraction_failed_count += 1
            if len(fail_samples) < 10:
                fail_samples.append({
                    'reason': 'no_answer_tag',
                    'idx': idx,
                    'model_response': model_response,
                    'input_prompt': input_prompt
                })
            continue
        answer = match.group(1)
        answer_norm = re.sub(r'[^\w]', '', answer).lower()
        if answer_norm not in ['yes', 'no']:
            fail_not_yes_no += 1
            extraction_failed_count += 1
            if len(fail_samples) < 10:
                fail_samples.append({
                    'reason': 'not_yes_no',
                    'idx': idx,
                    'answer': answer,
                    'model_response': model_response,
                    'input_prompt': input_prompt
                })
            continue

        # 检查是否是正确的回答
        if is_correct_response(is_match, answer_norm):
            correct_count += 1
            # 转换为所需格式
            filtered_item = {
                "messages": [
                    {
                        "role": "user",
                        "content": input_prompt
                    },
                    {
                        "role": "assistant", 
                        "content": model_response
                    }
                ]
            }
            filtered_data.append(filtered_item)
    
    # 保存过滤后的数据
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)

        print(f"处理完成!")
        print(f"原始样本数: {total_count}")
        print(f"提取失败数: {extraction_failed_count}")
        print(f"  - 缺少 <answer> 标签: {fail_no_answer_tag}")
        print(f"  - <answer> 内容不是yes/no: {fail_not_yes_no}")
        print(f"  - 缺失必要字段: {fail_missing_field}")
        print(f"  - model_response 为空: {fail_empty_response}")
        print(f"正确回答数: {correct_count}")
        print(f"过滤后样本数: {len(filtered_data)}")
        print(f"结果已保存到: {output_file}")
        if fail_samples:
            print("\n部分提取失败样本（最多10条）：")
            for i, item in enumerate(fail_samples):
                print(f"[{i+1}] 原因: {item['reason']}, 索引: {item['idx']}")
                if 'model_response' in item:
                    print(f"  model_response: {item['model_response'][:200]}")
                if 'answer' in item:
                    print(f"  answer: {item['answer']}")
                if 'input_prompt' in item:
                    print(f"  input_prompt: {item['input_prompt'][:100]}")
                if 'response' in item:
                    print(f"  response: {item['response']}")
                print("---")
        print()
    except Exception as e:
        print(f"错误: 无法保存文件 {output_file}: {e}")


def main():
    input_file = "/data/hongdeyao/code/RRec/sft/data/sft_deepseek_movies_response_10000.json"
    output_file = "/data/hongdeyao/code/RRec/sft/LLaMA-Factory/data/num_3000_yes1000_no2000_movies.json"
    
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    filter_correct_responses(input_file, output_file)


if __name__ == "__main__":
    main()
