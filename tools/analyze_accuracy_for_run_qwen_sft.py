#!/usr/bin/env python3
"""
读取json列表，输出准确率
"""

import json
import os
import re
import glob


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


def is_correct(is_match, answer):
    """判断预测是否正确"""
    if answer is None:  # 提取失败
        return None
    if is_match and answer == "yes":
        return True
    elif not is_match and answer == "no":
        return True
    return False


def analyze_single_file(file_path, exclude_sample_ids=None):
    """分析单个JSON文件，可选传入 sample_id 排除列表"""
    exclude_set = set(exclude_sample_ids) if exclude_sample_ids else set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}")
        return None

    if 'responses' not in data:
        print(f"警告: 文件 {file_path} 中没有 'responses' 字段")
        return None

    responses = data['responses']
    total = len(responses)
    valid_total = 0  # 有效样本数（提取成功的）
    correct = 0
    extraction_failed = 0
    failed_responses = []  # 存储提取失败的原始内容

    # 分组统计
    true_match_total = 0
    true_match_valid = 0
    true_match_correct = 0
    false_match_total = 0
    false_match_valid = 0
    false_match_correct = 0

    for i, response in enumerate(responses):
        sample_id = response.get('sample_id')
        if sample_id in exclude_set:
            continue  # 跳过排除的 sample_id
        is_match = response.get('is_match')
        model_response = response.get('model_response', '')

        answer = extract_answer(model_response)

        if answer is None:  # 提取失败
            extraction_failed += 1
            failed_responses.append({
                'index': i,
                'is_match': is_match,
                'model_response': model_response[:200] + '...' if len(model_response) > 200 else model_response
            })
            # 不计入准确率统计
            if is_match:
                true_match_total += 1
            else:
                false_match_total += 1
            continue

        # 提取成功的样本
        valid_total += 1
        correct_pred = is_correct(is_match, answer)

        if correct_pred:
            correct += 1

        if is_match:
            true_match_total += 1
            true_match_valid += 1
            if correct_pred:
                true_match_correct += 1
        else:
            false_match_total += 1
            false_match_valid += 1
            if correct_pred:
                false_match_correct += 1

    return {
        'file': os.path.basename(file_path),
        'total': total,
        'valid_total': valid_total,
        'extraction_failed': extraction_failed,
        'failed_responses': failed_responses,
        'correct': correct,
        'accuracy': correct / valid_total if valid_total > 0 else 0,
        'true_match_total': true_match_total,
        'true_match_valid': true_match_valid,
        'true_match_correct': true_match_correct,
        'true_match_accuracy': true_match_correct / true_match_valid if true_match_valid > 0 else 0,
        'false_match_total': false_match_total,
        'false_match_valid': false_match_valid,
        'false_match_correct': false_match_correct,
        'false_match_accuracy': false_match_correct / false_match_valid if false_match_valid > 0 else 0
    }


def main():
    # 手动设置json文件
    json_files = [
        "/data/hongdeyao/code/RRec/sft/data_rank/output_normal1.json",
        "/data/hongdeyao/code/RRec/sft/data_rank/output_merged1.json",
        "/data/hongdeyao/code/RRec/sft/data_rank/output_check1.json"
    ]

    # 需要排除的 sample_id 列表
    # exclude_sample_ids = [22, 75, 285, 910, 1205, 1422, 1487, 1562, 1739, 1756, 1835, 1937, 2095, 2118, 2409, 2438, 2799, 2884, 2909, 3230, 3410, 3615, 3753, 3863, 3919, 4007, 4961, 5055, 5134, 5427]
    exclude_sample_ids = []

    all_results = []

    # 分析每个文件
    for file_path in sorted(json_files):
        result = analyze_single_file(file_path, exclude_sample_ids=exclude_sample_ids)
        if result:
            all_results.append(result)

            print(f"文件: {result['file']}")
            print(f"  总样本: {result['total']}")
            print(f"  有效样本: {result['valid_total']} (成功提取answer)")
            print(f"  提取失败: {result['extraction_failed']}")
            print(f"  总体准确率: {result['accuracy']:.4f} ({result['correct']}/{result['valid_total']})")
            print(f"  is_match=True 准确率: {result['true_match_accuracy']:.4f} ({result['true_match_correct']}/{result['true_match_valid']})")
            print(f"  is_match=False 准确率: {result['false_match_accuracy']:.4f} ({result['false_match_correct']}/{result['false_match_valid']})")

            # 打印提取失败的样本
            if result['failed_responses']:
                print(f"  提取失败的样本 ({len(result['failed_responses'])}个):")
                for failed in result['failed_responses'][:5]:  # 只显示前5个
                    print(f"    样本 {failed['index']}: is_match={failed['is_match']}")
                    print(f"    原始内容: {failed['model_response']}")
                    print()
                if len(result['failed_responses']) > 5:
                    print(f"    ... 还有 {len(result['failed_responses']) - 5} 个提取失败的样本")
            print()
    
    # # 计算总体统计(暂时用不上)
    # if all_results:
    #     total_samples = sum(r['total'] for r in all_results)
    #     total_valid_samples = sum(r['valid_total'] for r in all_results)
    #     total_extraction_failed = sum(r['extraction_failed'] for r in all_results)
    #     total_correct = sum(r['correct'] for r in all_results)
    #     total_true_match_samples = sum(r['true_match_total'] for r in all_results)
    #     total_true_match_valid = sum(r['true_match_valid'] for r in all_results)
    #     total_true_match_correct = sum(r['true_match_correct'] for r in all_results)
    #     total_false_match_samples = sum(r['false_match_total'] for r in all_results)
    #     total_false_match_valid = sum(r['false_match_valid'] for r in all_results)
    #     total_false_match_correct = sum(r['false_match_correct'] for r in all_results)
        
    #     print("=" * 60)
    #     print("总体统计:")
    #     print(f"  文件数量: {len(all_results)}")
    #     print(f"  总样本数: {total_samples}")
    #     print(f"  有效样本数: {total_valid_samples}")
    #     print(f"  提取失败数: {total_extraction_failed} ({total_extraction_failed/total_samples:.2%})")
    #     print(f"  总体准确率: {total_correct/total_valid_samples:.4f} ({total_correct}/{total_valid_samples})")
    #     print(f"  is_match=True 总体准确率: {total_true_match_correct/total_true_match_valid:.4f} ({total_true_match_correct}/{total_true_match_valid})")
    #     print(f"  is_match=False 总体准确率: {total_false_match_correct/total_false_match_valid:.4f} ({total_false_match_correct}/{total_false_match_valid})")


if __name__ == "__main__":
    main()
