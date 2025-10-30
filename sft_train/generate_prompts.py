#!/usr/bin/env python3
"""
生成可配置匹配/不匹配比例的二分类prompt问题

主要功能：
1. 从Musical_Instruments数据集随机抽取n个样本
2. 可配置多少个prompt使用匹配的候选物品，多少个使用不匹配的候选物品
3. 使用固定随机种子42确保结果可复现

使用方法示例：
python generate_prompts.py --num_samples 100 --match_ratio 0.5 --dataset_dir /data/hongdeyao/Musical_Instruments_0_2022-10-2023-10
"""

import os
import sys
import json
import random
import numpy as np
from typing import List, Dict, Any
from datasets import load_from_disk
import argparse
from datetime import datetime

# 添加项目路径
sys.path.append('/data/hongdeyao/code/RRec')

from paths import model_names
from trainers.utils import get_tokenizer
from prompters.rrec_prompter import UserGenPrompter


class PromptGenerator:
    def __init__(self, 
                 dataset_dir: str,
                 model_type: str = "qwen",
                 seed: int = 42):
        """
        初始化prompt生成器
        """
        self.dataset_dir = dataset_dir
        self.model_type = model_type
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"🚀 Initializing Prompt Generator")
        print(f"📊 Dataset: {dataset_dir}")
        print(f"🤖 Model: {model_type}")
        print(f"🎲 Seed: {seed}")

        # 加载数据集
        self.load_dataset()
        
        # 加载tokenizer
        self.load_tokenizer()
        
        # 初始化prompter
        self.init_prompter()

    def load_dataset(self):
        """加载数据集"""
        print(f"📂 Loading dataset from {self.dataset_dir}")
        try:
            self.dataset = load_from_disk(self.dataset_dir)
            self.test_data = self.dataset['valid']
            self.item_info = self.dataset['item_info']
            # 创建item映射
            self.create_item_mappings()
            print(f"✅ Dataset loaded successfully:")
            print(f"  Test samples: {len(self.test_data)}")
            print(f"  Items: {len(self.item_info)}")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise

    def create_item_mappings(self):
        """创建物品ID到标题的映射"""
        self.id2title = {}
        self.all_item_ids = []
        self.item_dict = {}
        
        for item in self.item_info:
            item_id = item['item_id']
            title = item.get('title', f'Unknown Item {item_id}')
            self.id2title[item_id] = title
            self.item_dict[item_id] = item
            if item_id != 0:  # 排除padding项
                self.all_item_ids.append(item_id)
        
        print(f"📋 Created mappings for {len(self.id2title)} items (including {len(self.all_item_ids)} non-pad items)")

    def load_tokenizer(self):
        """加载tokenizer"""
        print(f"🔄 Loading tokenizer for model family: {self.model_type}")
        try:
            if self.model_type == 'qwen':
                model_name = model_names["Qwen2.5-3B-Instruct"]
            elif self.model_type == 'gemma':
                model_name = model_names["Gemma-2-2b-it"]
            else:
                model_name = list(model_names.values())[0]

            self.tokenizer = get_tokenizer(model_name)
            print("✅ Tokenizer loaded")
        except Exception as e:
            print(f"⚠️ Warning: failed to load tokenizer: {e}. Some prompt templating features may be limited.")
            self.tokenizer = None

    def init_prompter(self):
        """初始化prompter"""
        print("🔄 Initializing prompter...")
        
        # 根据数据集路径确定类别
        dataset_name = self.dataset_dir.split('/')[-1]
        if 'Musical_Instruments' in dataset_name:
            category = 'Musical_Instruments'
        elif 'Video_Games' in dataset_name:
            category = 'Video_Games'
        elif 'CDs_and_Vinyl' in dataset_name:
            category = 'CDs_and_Vinyl'
        else:
            category = 'Musical_Instruments'
            print(f"⚠️ Could not determine category from {dataset_name}, using default: {category}")

        window_size = 20
        emb_token = '<answer>'
        emb_end_token = '</answer>'

        full_dataset = {
            'test': self.test_data,
            'item_info': self.item_info
        }

        self.user_gen_prompter = UserGenPrompter(
            dset=full_dataset,
            tokenizer=self.tokenizer,
            category=category,
            window_size=window_size,
            emb_token=emb_token,
            emb_end_token=emb_end_token
        )

        print("✅ Prompter initialized")

    def build_judgment_prompt(self, user_prompt: str, candidate_item_id: int) -> str:
        """
        构建判断prompt，与deepseek评估代码中的格式保持一致
        """
        # 获取候选物品信息
        item = self.item_dict.get(candidate_item_id, {})
        title = item.get('title', f'Unknown Item {candidate_item_id}')
        description = item.get('description', '')
        if isinstance(description, list):
            description = ' '.join(description[::-1]) if description else ''
        avg_rating = item.get('average_rating', '')
        num_buyers = item.get('rating_number', '')

        # 构建判断指令
        
        # check prompt
        judgment_instruction = (
            "Please judge how well the candidate item matches the user's preferences based on the history below and the candidate item information. "
            "Produce a **single coherent paragraph** that contains the following three parts **in order**: initial reasoning, self-check/evaluation, and a final reasoning. "
            "**Mark each part by placing a single `#` immediately before its label** (for example `#Initial reasoning:`, `#Self-check:`, `#Final conclusion and suggestion:`). "
            "Do not add other `#` symbols. Please keep the entire paragraph to **roughly 400–800 words**. "
            "It is acceptable to draw reasonable inferences from the provided information. "
            "Finally, conclude with a clear \"yes\" or \"no\" judgment indicating whether the candidate item matches the user's preferences, enclosed within `<answer>` and `</answer>`.\n\n"
            "* `#Initial reasoning:` — perform a normal, thorough reasoning step that **fully considers relevant factors**.\n"
            "* `#Self-check:` — evaluate the initial reasoning: explicitly identify any mistakes, overlooked evidence, uncertainties, or over-weighted signals in the first pass; either confirm the initial reasoning or explain why and how you would modify it. This is a short critical reflection on the first pass.\n"
            "* `#Final conclusion and suggestion:` — combine the outcomes of the first two parts and present a succinct final reasoning and conclusion.\n"
        )
        
        # judgment_instruction = (
        #     f"""
        #     Please judge how well the candidate item matches the user's preferences based on the history below and the candidate item information. Produce a single coherent paragraph that contains the following two parts in order, each marked by a single `#` immediately before its label (for example `#Hypotheses:` and `#Weighing and conclusion:`). Do not add any other `#` symbols. Keep the paragraph roughly 200–400 words. It is acceptable to draw reasonable inferences from the provided information. Finally, conclude with a clear “yes” or “no” judgment indicating whether the candidate item matches the user’s preferences, enclosed within `<answer>` and `</answer>`.
        #     #Hypotheses: List 2–4 distinct and competing hypotheses (H1, H2, ...) about the user's preference/intent regarding this item. **You MUST include at least one hypothesis that supports a match and one that challenges it.** For each hypothesis, give a one-sentence explanation and cite the specific evidence from HISTORY or CANDIDATE ITEM that supports it (format each as `H1: ... — evidence: ...; H2: ... — evidence: ...;`).
        #     #Weighing and conclusion: Weigh the strength of the evidence for and against each hypothesis. **Explicitly compare the supporting and challenging evidence, explaining which is more definitive and why.** State which hypothesis you consider most likely and **justify this choice by acknowledging the limitations or strength of the counter-evidence.** Finally, give a final match judgement that reflects this nuanced weighing.
        #     """
        # )
        
        # normal prompt
        # judgment_instruction = (
        #     "Please judge how well the candidate item matches the user's preferences based on the history below and the candidate item information. "
        #     "Finally, conclude with a clear \"yes\" or \"no\" judgment indicating whether the candidate item matches the user's preferences, enclosed within `<answer>` and `</answer>`.\n\n"
        # )
        
        # 只保留 "purchases and ratings (out of 5):\n" 后面的部分
        split_str = "purchases and ratings (out of 5):\n"
        if split_str in user_prompt:
            user_prompt = user_prompt.split(split_str, 1)[-1].strip()

        prompt = (
            f"{judgment_instruction}\n\n"
            f"# HISTORY:\n\n"
            f"historical musical instruments purchases and ratings (out of 5):\n{user_prompt}\n\n"
            f"# CANDIDATE ITEM:\n\n"
            f"Title: {title}\n"
            f"Average Rating: {avg_rating}\n"
            f"Number of Ratings: {num_buyers}\n"
            f"Description: {description}\n\n"
            f"# TASK:\n\n"
            f"Based on the user's history above, produce the single-paragraph output described (with parts labeled by a single `#` as explained), followed by a final yes/no judgment within `<answer>` tags.\n"
            # f"Based on the user's history above, produce the single-paragraph output described, followed by a final yes/no judgment within `<answer>` tags.\n"
        )
        
        return prompt

    def generate_prompts(self, 
                        num_samples: int = 100, 
                        match_ratio: float = 0.5) -> List[Dict[str, Any]]:
        """
        生成指定数量的prompt问题
        
        Args:
            num_samples: 生成的总样本数
            match_ratio: 匹配样本的比例（0.0-1.0）
        
        Returns:
            包含prompt信息的字典列表
        """
        print(f"\n🎯 Generating {num_samples} prompts with {match_ratio:.1%} match ratio")
        print("=" * 60)
        
        # 计算匹配和不匹配的数量
        num_match = int(num_samples * match_ratio)
        num_no_match = num_samples - num_match
        
        print(f"📊 Distribution:")
        print(f"  Match samples: {num_match}")
        print(f"  No-match samples: {num_no_match}")
        
        prompts = []
        
        # 随机打乱测试数据索引
        test_indices = list(range(len(self.test_data)))
        random.shuffle(test_indices)
        
        # 确保有足够的样本
        if len(test_indices) < num_samples:
            print(f"⚠️ Warning: Requested {num_samples} samples but only {len(test_indices)} available")
            num_samples = len(test_indices)
            num_match = int(num_samples * match_ratio)
            num_no_match = num_samples - num_match
        
        # 创建样本类型列表（True表示匹配，False表示不匹配）
        sample_types = [True] * num_match + [False] * num_no_match
        random.shuffle(sample_types)  # 随机打乱顺序
        
        valid_prompts = 0
        
        for i, is_match in enumerate(sample_types):
            if valid_prompts >= num_samples:
                break
                
            # 获取测试样本
            sample_idx = test_indices[i]
            sample = self.test_data[sample_idx]
            
            history_ids = sample['history_item_id']
            target_id = sample['item_id']
            
            # 跳过无效样本
            if target_id == 0:
                continue
                
            history_ids = [id for id in history_ids if id != 0]
            history_titles = [self.id2title.get(id, f"Unknown_{id}") for id in history_ids]
            target_title = self.id2title.get(target_id, f"Unknown_{target_id}")
            
            # 获取可用的候选物品（不在历史记录中）
            available_items = [item_id for item_id in self.all_item_ids if item_id not in history_ids]
            if target_id not in available_items:
                print(f"Warning: Target item {target_id} is in history for sample {sample_idx}")
                continue
            
            # 根据is_match选择候选物品
            if is_match:
                # 匹配：使用目标物品
                candidate_id = target_id
                true_label = True
            else:
                # 不匹配：随机选择非目标物品
                wrong_candidates = [item_id for item_id in available_items if item_id != target_id]
                if not wrong_candidates:
                    print(f"Warning: No wrong candidates available for sample {sample_idx}")
                    continue
                candidate_id = random.choice(wrong_candidates)
                true_label = False
            
            # 生成用户prompt
            sample_copy = dict(sample)
            sample_copy['seq_input_ids'] = sample_copy['history_item_id']
            sample_copy['seq_labels'] = sample_copy['item_id']
            
            try:
                chat_example = self.user_gen_prompter.to_chat_example(sample_copy)
                user_prompt = chat_example.get('prompt', '')
            except Exception as e:
                print(f"Warning: Failed to generate user prompt for sample {sample_idx}: {e}")
                continue
            
            # 构建完整的判断prompt
            full_prompt = self.build_judgment_prompt(user_prompt, candidate_id)
            
            # 获取候选物品标题
            candidate_title = self.id2title.get(candidate_id, f"Unknown_{candidate_id}")
            
            # 创建prompt字典
            prompt_dict = {
                'prompt_id': valid_prompts,
                'sample_id': sample_idx,
                'user_id': sample.get('user_id', 'unknown'),
                'history_item_ids': history_ids[-10:],  # 只保留最近10个
                'history_titles': history_titles[-10:],
                'target_item_id': target_id,
                'target_title': target_title,
                'candidate_item_id': candidate_id,
                'candidate_title': candidate_title,
                'is_match': is_match,
                'user_prompt': user_prompt,
                'full_prompt': full_prompt,
                'timestamp': datetime.now().isoformat()
            }
            
            prompts.append(prompt_dict)
            valid_prompts += 1
            
            if valid_prompts % 10 == 0:
                print(f"✅ Generated {valid_prompts}/{num_samples} prompts")
        
        print(f"\n🎉 Successfully generated {len(prompts)} prompts")
        return prompts

    def save_prompts(self, prompts: List[Dict[str, Any]], output_file: str):
        """保存prompts到文件"""
        print(f"💾 Saving {len(prompts)} prompts to {output_file}")
        
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 准备保存的数据
        save_data = {
            'metadata': {
                'total_prompts': len(prompts),
                'dataset_dir': self.dataset_dir,
                'model_type': self.model_type,
                'seed': self.seed,
                'generation_time': datetime.now().isoformat(),
                'match_count': sum(1 for p in prompts if p['is_match']),
                'no_match_count': sum(1 for p in prompts if not p['is_match'])
            },
            'prompts': prompts
        }
        
        # 保存为JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Prompts saved successfully!")
        print(f"📊 Summary:")
        print(f"  Total prompts: {save_data['metadata']['total_prompts']}")
        print(f"  Match prompts: {save_data['metadata']['match_count']}")
        print(f"  No-match prompts: {save_data['metadata']['no_match_count']}")


def main():
    parser = argparse.ArgumentParser(description="Generate configurable match/no-match binary classification prompts")
    parser.add_argument("--dataset_dir", type=str, 
                       default="/data/hongdeyao/Movies_and_TV_0_2022-10-2023-10",
                       help="Dataset directory")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of prompts to generate")
    parser.add_argument("--match_ratio", type=float, default=0.5,
                       help="Ratio of match samples (0.0-1.0)")
    parser.add_argument("--model_type", type=str, default="qwen",
                       choices=["qwen", "gemma"],
                       help="Model type for prompt formatting")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--output_file", type=str, 
                       default="/data/hongdeyao/code/RRec/sft/data/sft_train_movies_prompts_10000.json",
                       help="Output file path")

    args = parser.parse_args()

    # 验证参数
    if not 0.0 <= args.match_ratio <= 1.0:
        print("❌ Error: match_ratio must be between 0.0 and 1.0")
        return
    
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Error: Dataset directory not found: {args.dataset_dir}")
        return

    # 创建生成器
    generator = PromptGenerator(
        dataset_dir=args.dataset_dir,
        model_type=args.model_type,
        seed=args.seed
    )

    # 生成prompts
    prompts = generator.generate_prompts(
        num_samples=args.num_samples,
        match_ratio=args.match_ratio
    )

    # 保存prompts
    generator.save_prompts(prompts, args.output_file)

    print("\n🎉 Prompt generation completed successfully!")


if __name__ == "__main__":
    main()
