import os
import json
import numpy as np
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import time
import argparse
from openai import OpenAI
import random
import re

# --test_only

class APIRecommendationTester:
    VALID_TARGET_POSITIONS = {"first", "last", "random"}

    def __init__(self, 
                 api_key: str, 
                 dataset_dir: str,
                 api_base: str = "http://115.182.62.174:18888/v1",
                 model_name: str = "deepseek-reasoner",
                 seed: int = 42,
                 target_position: str = "first",
                 output_dir: str = ".",
                 test_limit: Optional[int] = None,
                 predefined_candidates_path: Optional[str] = None,
                 predefined_candidate_split: str = "test"):
        """
        初始化API测试器，与训练脚本完全一致的评估方式
        
        Args:
            api_key: API密钥
            dataset_dir: 数据集目录路径
            api_base: API基础URL
            model_name: 使用的模型名称
            seed: 随机种子
            target_position: 正确候选在候选列表中的位置策略（first/last/random）
            output_dir: 结果文件保存目录
            test_limit: test-only模式下评估的最大样本数量（按数据顺序）
        """
        self.api_key = api_key
        self.dataset_dir = dataset_dir
        self.api_base = api_base
        self.model_name = model_name
        self.seed = seed
        self.target_position_mode = target_position.lower()
        if self.target_position_mode not in self.VALID_TARGET_POSITIONS:
            raise ValueError(f"target_position must be one of {sorted(self.VALID_TARGET_POSITIONS)}, got {target_position}")
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.test_limit = test_limit
        self.test_only_mode = False
        self.predefined_candidate_split = predefined_candidate_split.lower() if predefined_candidate_split else "test"
        self.predefined_candidates_path = predefined_candidates_path
        self.predefined_candidate_entries: Optional[List[Dict[str, Any]]] = None
        self.predefined_candidate_config: Dict[str, Any] = {}
        self._split_user_index_cache: Dict[str, Dict[str, List[int]]] = {}
        
        # 初始化OpenAI客户端
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        
        # 加载数据集
        self.load_dataset()

        if self.predefined_candidates_path:
            self.load_predefined_candidates(self.predefined_candidates_path)

    def prepare_rank_candidates(self, 
                                 data_split: List[Dict[str, Any]],
                                 num_users: Optional[int],
                                 split_name: str,
                                 preserve_order: bool = False) -> List[Dict[str, Any]]:
        """使用与run_rrec.py和generate_prompts_rank.py一致的方式构建候选集合"""
        if self.predefined_candidate_entries and split_name == self.predefined_candidate_split:
            return self.prepare_rank_candidates_from_predefined(data_split, num_users, split_name)
        order_note = "preserving dataset order" if preserve_order else "with shuffled users"
        print(f"\n🔄 Preparing rank candidates for {split_name} split using training-consistent sampler ({order_note})")

        user_samples: Dict[Any, List[Dict[str, Any]]] = {}
        for idx in range(len(data_split)):
            sample = data_split[idx]
            user_id = sample.get('user_id')
            if user_id is None:
                continue
            if user_id not in user_samples:
                user_samples[user_id] = []
            user_samples[user_id].append({'index': idx, 'sample': sample})

        available_users = list(user_samples.keys())
        if not preserve_order:
            random.shuffle(available_users)

        if not available_users:
            print(f"❌ Error: No valid users found in {split_name} split")
            return []

        requested_users = num_users if num_users is not None else len(available_users)
        if requested_users > len(available_users):
            print(f"⚠️ Warning: Requested {requested_users} users but only {len(available_users)} available in {split_name}, adjusting")
            requested_users = len(available_users)

        prepared_samples: List[Dict[str, Any]] = []

        for user_id in available_users:
            if len(prepared_samples) >= requested_users:
                break

            entries = user_samples[user_id]
            selected_entry: Optional[Dict[str, Any]] = None
            for entry in entries:
                target_id = entry['sample'].get('item_id', 0)
                if target_id and target_id in self.id2title:
                    selected_entry = entry
                    break

            if selected_entry is None:
                print(f"Warning: No valid sample found for user {user_id}, skipping")
                continue

            sample_idx = selected_entry['index']
            sample = selected_entry['sample']

            history_ids = [hid for hid in sample['history_item_id'] if hid != 0]
            # if len(history_ids) < 5:
            #     print(f"Warning: User {user_id} has only {len(history_ids)} history items, skipping")
            #     continue

            target_id = sample['item_id']
            if target_id == 0:
                print(f"Warning: Sample {sample_idx} for user {user_id} has padding target, skipping")
                continue

            available_items = [item_id for item_id in self.all_item_ids if item_id not in history_ids]
            if target_id not in available_items:
                available_items.append(target_id)

            wrong_pool = [item_id for item_id in available_items if item_id != target_id]
            if len(wrong_pool) < 19:
                print(f"Warning: User {user_id} only has {len(wrong_pool)} wrong candidates available, skipping")
                continue

            wrong_candidates = random.sample(wrong_pool, 19)
            candidate_items = wrong_candidates.copy()

            if self.target_position_mode == "first":
                candidate_items.insert(0, target_id)
            elif self.target_position_mode == "last":
                candidate_items.append(target_id)
            else:
                insert_idx = random.randint(0, len(candidate_items))
                candidate_items.insert(insert_idx, target_id)

            history_titles = [self.id2title.get(hid, f"Unknown_{hid}") for hid in history_ids]
            target_title = self.id2title.get(target_id, f"Unknown_{target_id}")

            prepared_samples.append({
                'sample_index': sample_idx,
                'sample': sample,
                'user_id': user_id,
                'history_ids': history_ids,
                'history_titles': history_titles,
                'target_id': target_id,
                'target_title': target_title,
                'candidate_items': candidate_items,
                'target_candidate_position': candidate_items.index(target_id) + 1
            })

            print(f"✅ Prepared candidate set for user {user_id} (sample {sample_idx}) in {split_name}")

        if len(prepared_samples) < requested_users:
            print(f"⚠️ Warning: Only prepared {len(prepared_samples)} user samples (requested {requested_users}) for {split_name}")

        return prepared_samples

    def prepare_rank_candidates_from_predefined(self,
                                                data_split: List[Dict[str, Any]],
                                                num_users: Optional[int],
                                                split_name: str) -> List[Dict[str, Any]]:
        """使用预定义的JSON候选集合来构建评估样本"""
        print(f"\n🗂️  Preparing rank candidates for {split_name} split using predefined JSON entries (preserving file order)")

        if not self.predefined_candidate_entries:
            print("⚠️  Warning: No predefined entries available; returning empty list")
            return []

        requested_users = num_users if num_users is not None else len(self.predefined_candidate_entries)
        if requested_users > len(self.predefined_candidate_entries):
            print(f"⚠️  Warning: Requested {requested_users} entries but only {len(self.predefined_candidate_entries)} available; adjusting")
            requested_users = len(self.predefined_candidate_entries)

        user_index_map = self._build_split_user_index(data_split, split_name)

        prepared_samples: List[Dict[str, Any]] = []

        for entry_idx, entry in enumerate(self.predefined_candidate_entries[:requested_users]):
            sample_index_raw = entry.get('sample_index')
            user_id = entry.get('user_id')
            target_id_raw = entry.get('target_item_id')
            candidate_ids = entry.get('candidate_ids') or []

            if not candidate_ids:
                print(f"⚠️  Warning: Entry #{entry_idx} missing candidate IDs, skipping")
                continue

            sample_index: Optional[int] = None
            if sample_index_raw is not None:
                try:
                    sample_index = int(sample_index_raw)
                except (TypeError, ValueError):
                    sample_index = None

            target_id: Optional[int] = None
            if target_id_raw is not None:
                try:
                    target_id = int(target_id_raw)
                except (TypeError, ValueError):
                    target_id = None

            sample = None
            resolved_sample_index: Optional[int] = None

            if sample_index is not None and 0 <= sample_index < len(data_split):
                candidate_sample = data_split[sample_index]
                if user_id is None or candidate_sample.get('user_id') == user_id:
                    sample = candidate_sample
                    resolved_sample_index = sample_index

            if sample is None and user_id is not None:
                candidate_indices = user_index_map.get(user_id, [])
                for idx in candidate_indices:
                    candidate_sample = data_split[idx]
                    if target_id is None or candidate_sample.get('item_id') == target_id:
                        sample = candidate_sample
                        resolved_sample_index = idx
                        break
                if sample is None and candidate_indices:
                    sample = data_split[candidate_indices[0]]
                    resolved_sample_index = candidate_indices[0]

            if sample is None:
                print(f"⚠️  Warning: Unable to locate dataset sample for entry #{entry_idx} (user_id={user_id}, sample_index={sample_index}), skipping")
                continue

            history_ids = [hid for hid in sample['history_item_id'] if hid != 0]
            history_titles = [self.id2title.get(hid, f"Unknown_{hid}") for hid in history_ids]
            dataset_target_id = sample.get('item_id', 0)

            if target_id in (None, 0):
                target_id = dataset_target_id

            if target_id == 0:
                print(f"⚠️  Warning: Entry #{entry_idx} has invalid target item (0), skipping")
                continue

            candidate_items = list(dict.fromkeys(candidate_ids))  # 去重且保持顺序
            if target_id not in candidate_items:
                candidate_items.append(target_id)
                print(f"ℹ️  Info: Target item {target_id} not in candidate list for entry #{entry_idx}; appended to the end")

            target_title = self.id2title.get(target_id, f"Unknown_{target_id}")

            try:
                target_candidate_position = candidate_items.index(target_id) + 1
            except ValueError:
                target_candidate_position = None

            prepared_samples.append({
                'sample_index': resolved_sample_index if resolved_sample_index is not None else -1,
                'sample': sample,
                'user_id': user_id or sample.get('user_id'),
                'history_ids': history_ids,
                'history_titles': history_titles,
                'target_id': target_id,
                'target_title': target_title,
                'candidate_items': candidate_items,
                'target_candidate_position': target_candidate_position
            })

            print(f"✅ Prepared JSON candidate set for user {user_id or sample.get('user_id')} (entry #{entry_idx}, dataset index {resolved_sample_index})")

        if len(prepared_samples) < requested_users:
            print(f"⚠️  Warning: Only prepared {len(prepared_samples)} JSON-defined samples (requested {requested_users}) for {split_name}")

        return prepared_samples
        
    def load_dataset(self):
        """加载数据集，与训练脚本完全一致"""
        print(f"Loading dataset from {self.dataset_dir}")
        try:
            self.dataset = load_from_disk(self.dataset_dir)
            self.test_data = self.dataset['test']
            self.valid_data = self.dataset['valid']
            self.train_data = self.dataset['train']
            self.item_info = self.dataset['item_info']
            
            # 创建item映射，与训练脚本一致
            self.create_item_mappings()
            
            print(f"Dataset loaded successfully:")
            print(f"  Train samples: {len(self.train_data)}")
            print(f"  Valid samples: {len(self.valid_data)}")
            print(f"  Test samples: {len(self.test_data)}")
            print(f"  Items: {len(self.item_info)}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def create_item_mappings(self):
        """创建商品映射，与训练脚本完全一致"""
        self.id2title = {}
        self.id2asin = {}
        self.asin2title = {}
        self.all_item_ids = []
        
        for item in self.item_info:
            item_id = item['item_id']
            title = item.get('title', f'Unknown Item {item_id}')
            asin = item.get('parent_asin', f'unknown_asin_{item_id}')
            
            self.id2title[item_id] = title
            self.id2asin[item_id] = asin
            self.asin2title[asin] = title
            
            if item_id != 0:  # 排除padding token，与训练脚本一致
                self.all_item_ids.append(item_id)
        
        print(f"Created mappings for {len(self.id2title)} items (including {len(self.all_item_ids)} non-pad items)")

    def load_predefined_candidates(self, path: str) -> None:
        """加载预定义的候选集合，用于JSON模式的数据抽取"""
        resolved_path = os.path.abspath(path)
        if not os.path.exists(resolved_path):
            print(f"⚠️  Warning: Predefined candidate file not found at {resolved_path}. Falling back to random sampling.")
            self.predefined_candidates_path = None
            return

        try:
            with open(resolved_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            print(f"⚠️  Warning: Failed to load predefined candidate file: {exc}. Falling back to random sampling.")
            self.predefined_candidates_path = None
            return

        entries = []
        if isinstance(data, dict):
            self.predefined_candidate_config = data.get('config', {})
            entries = data.get('detailed_results') or []
        elif isinstance(data, list):
            entries = data
        else:
            entries = []

        if not isinstance(entries, list) or not entries:
            print("⚠️  Warning: Predefined candidate file does not contain usable entries. Falling back to random sampling.")
            self.predefined_candidates_path = None
            return

        # 规范化字段，确保候选ID为整数
        normalized_entries: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            original_candidate_ids = entry.get('candidate_ids') or []
            try:
                candidate_ids: List[int] = []
                for item in original_candidate_ids:
                    value = int(item)
                    if value != 0:
                        candidate_ids.append(value)
            except Exception:
                # 如果转换失败则跳过该条目
                continue
            normalized_entries.append({
                **entry,
                'candidate_ids': candidate_ids
            })

        if not normalized_entries:
            print("⚠️  Warning: No valid entries found in predefined candidate file. Falling back to random sampling.")
            self.predefined_candidates_path = None
            return

        self.predefined_candidate_entries = normalized_entries
        total_entries = len(self.predefined_candidate_entries)
        print(f"📄 Loaded {total_entries} predefined candidate entries from {resolved_path}")

        # 校验数据集路径是否匹配
        expected_dataset = self.predefined_candidate_config.get('dataset_path')
        if expected_dataset and os.path.normpath(expected_dataset) != os.path.normpath(self.dataset_dir):
            print(f"⚠️  Warning: Candidate file dataset ({expected_dataset}) does not match current dataset ({self.dataset_dir})")

    def _build_split_user_index(self, data_split: List[Dict[str, Any]], split_name: str) -> Dict[str, List[int]]:
        """缓存指定划分下user_id到样本索引的映射，加速查找"""
        cache = self._split_user_index_cache.get(split_name)
        if cache is not None:
            return cache

        mapping: Dict[str, List[int]] = {}
        for idx in range(len(data_split)):
            sample = data_split[idx]
            user_id = sample.get('user_id')
            if user_id is None:
                continue
            mapping.setdefault(user_id, []).append(idx)

        self._split_user_index_cache[split_name] = mapping
        return mapping
    
    def call_api(self, prompt: str, max_retries: int = 5) -> tuple:
        """
        调用API，使用提供的接口方式
        
        Args:
            prompt: 输入提示
            max_retries: 最大重试次数
            
        Returns:
            tuple: (API响应内容, 思考过程, 原始响应字典)
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful recommendation system assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=3000
                )
                
                # 获取响应内容和思考过程
                content = response.choices[0].message.content or ""
                reasoning = getattr(response.choices[0].message, 'reasoning', None) or ""
                
                # 转换为字典用于调试
                raw_response = response.model_dump() if hasattr(response, 'model_dump') else str(response)
                
                # 如果content为空，打印原始响应并重试
                if not content:
                    print(f"⚠️ WARNING: Empty response content received (attempt {attempt + 1}/{max_retries})!")
                    print(f"Raw response: {json.dumps(raw_response, indent=2, ensure_ascii=False) if isinstance(raw_response, dict) else raw_response}")
                    
                    if attempt < max_retries - 1:
                        print(f"🔄 Retrying due to empty content...")
                        time.sleep(2 ** attempt)  # 指数退避
                        continue  # 重试
                    else:
                        print(f"❌ All {max_retries} attempts returned empty content!")
                        return "", reasoning, raw_response
                
                # content不为空，返回成功结果
                return content, reasoning, raw_response
                
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    return "", "", {"error": str(e)}
    
    def create_full_ranking_prompt(self, 
                                 history_titles: List[str], 
                                 all_available_items: List[int],
                                 category: str) -> str:
        """
        创建全库商品排序的提示，与训练脚本的全库评估完全一致
        
        Args:
            history_titles: 用户历史商品标题
            all_available_items: 所有可用商品ID（排除历史商品）
            category: 商品类别
        """
        # 处理历史序列，与训练时的window_size=20保持一致
        history_display = history_titles[-20:] if len(history_titles) > 20 else history_titles
        
        # 构建历史序列字符串
        if len(history_display) == 0:
            history_str = "No previous purchases"
        else:
            history_str = " -> ".join(history_display)
        
        # 由于全库商品太多，我们随机采样一部分进行排序（模拟训练时的计算约束）
        # 但保证目标商品一定在其中
        sample_size = min(100, len(all_available_items))  # 与训练时的batch size类似
        
        prompt = f"""You are a recommendation system for {category} products.

User's purchase history: {history_str}

Based on this history, I need you to rank {sample_size} candidate items by likelihood of purchase.

Please provide your top 20 recommendations with item numbers, ranked from most likely to least likely.
Format your response as a comma-separated list of numbers only.

Example response: 15,3,7,42,88,12,55,73,9,21,34,67,81,96,11,45,78,23,56,39
"""
        
        return prompt
    
    def parse_ranking_response(self, response: str, candidate_items: List[int]) -> List[int]:
        """
        解析排序响应，返回按预测概率排序的商品ID
        
        Args:
            response: API响应
            candidate_items: 候选商品ID列表
            
        Returns:
            按预测概率排序的商品ID列表（只返回模型预测的top-K，不补充）
        """
        try:
            # 优先查找 RANKING: 开头的行
            ranking_line = None
            lines = response.strip().split('\n')
            
            # 方法1：查找 RANKING: 开头的行
            for line in lines:
                if line.strip().startswith('RANKING:'):
                    ranking_line = line.strip()
                    break
            
            if ranking_line:
                # 提取 RANKING: 后面的数字
                ranking_part = ranking_line.split('RANKING:', 1)[1].strip()
                numbers = re.findall(r'\b\d+\b', ranking_part)
                print(f"Found RANKING line: {ranking_part}") if len(candidate_items) <= 100 else None
            else:
                # 方法2：查找最后一行包含逗号分隔数字的行
                for line in reversed(lines):
                    line = line.strip()
                    if ',' in line and len(re.findall(r'\b\d+\b', line)) >= 10:
                        numbers = re.findall(r'\b\d+\b', line)
                        print(f"Found comma-separated line: {line}") if len(candidate_items) <= 100 else None
                        break
                else:
                    # 方法3：提取所有数字作为备选（最后的fallback）
                    numbers = re.findall(r'\b\d+\b', response)
                    print(f"Fallback: using all numbers from response") if len(candidate_items) <= 100 else None
            
            # 将数字映射回商品ID（只处理模型输出的数量，不补充）
            predicted_item_ids = []
            invalid_indices = []
            
            for i, num_str in enumerate(numbers):
                try:
                    idx = int(num_str) - 1  # 转换为0-indexed
                    if 0 <= idx < len(candidate_items):
                        item_id = candidate_items[idx]
                        if item_id not in predicted_item_ids:
                            predicted_item_ids.append(item_id)
                    else:
                        invalid_indices.append(num_str)
                except:
                    invalid_indices.append(num_str)
            
            if invalid_indices and len(candidate_items) <= 100:
                print(f"Invalid indices found: {invalid_indices[:5]}...")
            
            # 关键修改：不再补充剩余商品！
            # 只返回模型实际预测的排序，通常是top-20
            if len(candidate_items) <= 100:
                print(f"Successfully parsed {len(predicted_item_ids)} items (model prediction), first 10: {[candidate_items.index(id)+1 for id in predicted_item_ids[:10]]}")
            
            return predicted_item_ids
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            # 如果解析失败，返回空列表而不是随机排序
            return []
    
    def calculate_metrics_like_training(self, predicted_ranking: List[int], target_id: int) -> Dict[str, float]:
        """
        计算评估指标，与训练脚本完全一致
        采用与trainers/utils.py中calculate_metrics函数相同的逻辑
        
        Args:
            predicted_ranking: 预测的商品ID排序（可能只有top-K个，如20个）
            target_id: 目标商品ID
            
        Returns:
            评估指标字典
        """
        metrics = {}
        
        # 找到目标商品在预测排序中的位置（0-indexed，与训练脚本一致）
        try:
            target_position = predicted_ranking.index(target_id)  # 0-indexed position
        except ValueError:
            # 目标商品不在预测中，认为排在预测列表之后
            # 对于NDCG@K计算，如果目标不在top-K中，则贡献为0
            target_position = len(predicted_ranking)  # 设为预测列表长度，表示不在预测中
        
        # 计算Hit Rate @ K，与训练脚本的ks=[5, 10, 20]一致
        for k in [5, 10, 20]:
            # 如果目标在top-k中，则命中
            metrics[f'hit_rate@{k}'] = 1.0 if target_position < k else 0.0
        
        # 计算DCG @ K，使用与训练脚本完全相同的公式
        # 训练脚本：discount = torch.log2(torch.arange(2, cutoff + 2))
        # dcg = (1.0 / discount)
        # 注意：position从0开始，所以discount的索引是position+2
        for k in [5, 10, 20]:
            if target_position < k:
                # 与训练脚本完全一致：position 0对应log2(2), position 1对应log2(3)...
                dcg_value = 1.0 / np.log2(target_position + 2)
                metrics[f'ndcg@{k}'] = dcg_value  # 训练脚本没有IDCG归一化，直接使用DCG
            else:
                # 目标商品不在top-k中，贡献为0
                metrics[f'ndcg@{k}'] = 0.0
        
        return metrics
    
    def evaluate_like_training(self, 
                             num_samples: Optional[int] = None,  # 改为可选，默认测试全集
                             split: str = 'test',
                             save_results: bool = True,
                             progress_interval: int = 100,  # 新增：进度输出间隔
                             save_interval: int = 50,  # 新增：保存间隔
                             preserve_order: bool = False) -> Dict[str, float]:
        """
        评估函数，完全模拟训练脚本中的评估过程
        
        Args:
            num_samples: 测试样本数量，None表示测试全集（与训练脚本一致）
            split: 数据集分割 ('test' 或 'valid')
            save_results: 是否保存详细结果
            progress_interval: 每隔多少个样本输出一次进度（默认100）
            save_interval: 每隔多少个样本保存一次中间结果（默认50）
            
        Returns:
            评估指标字典
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        data = self.test_data if split == 'test' else self.valid_data
        total_samples = len(data)
        using_predefined = bool(self.predefined_candidate_entries) and split == self.predefined_candidate_split
        if using_predefined:
            predefined_total = len(self.predefined_candidate_entries)
            if num_samples is None or num_samples > predefined_total:
                num_samples = predefined_total
            print(f"ℹ️  Using predefined candidate JSON for {split} split with {predefined_total} available entries")
        
        # 如果num_samples为None，测试全集（与训练脚本一致）
        if num_samples is None:
            num_samples = total_samples
            print(f"\n=== Evaluating API like Training Script on {split} set (FULL DATASET: {total_samples} samples) ===")
        else:
            num_samples = min(num_samples, total_samples)
            print(f"\n=== Evaluating API like Training Script on {split} set ({num_samples}/{total_samples} samples, preserve_order={preserve_order}) ===")
        
        print(f"💾 Incremental save every {save_interval} samples")
        category = self.dataset_dir.split('/')[-1].split('_')[0].replace('_', ' ')

        prepared_samples = self.prepare_rank_candidates(data, num_samples, split, preserve_order=preserve_order)
        if not prepared_samples:
            print(f"❌ No valid user samples available for {split} evaluation")
            return {}

        total_to_evaluate = len(prepared_samples)
        candidate_count = len(prepared_samples[0]['candidate_items']) if prepared_samples else 0
        
        # 初始化累积指标，与训练脚本的MetricUpdater一致
        accumulated_metrics = {
            'hit_rate@5': 0.0, 'hit_rate@10': 0.0, 'hit_rate@20': 0.0,
            'ndcg@5': 0.0, 'ndcg@10': 0.0, 'ndcg@20': 0.0
        }
        
        valid_samples = 0
        detailed_results = []
        prompt_response_logs = []  # 新增：保存prompt和response用于调试
        
        for prepared in tqdm(prepared_samples, desc=f"Evaluating {split}"):
            sample = prepared['sample']
            history_ids = prepared['history_ids']
            history_titles = prepared['history_titles']
            target_id = prepared['target_id']
            target_title = prepared['target_title']
            candidate_items = prepared['candidate_items']
            sample_index = prepared['sample_index']
            user_id = prepared['user_id']
            target_candidate_position = prepared.get('target_candidate_position')
            
            # 构建候选商品标题列表用于prompt
            candidate_titles = [self.id2title[item_id] for item_id in candidate_items]
            candidates_str = "\n".join([f"{i+1}. {title}" for i, title in enumerate(candidate_titles)])
            
            # 创建prompt，明确指定输出格式
            prompt = f"""You are a recommendation system for {category} products.

User's purchase history: {" -> ".join(history_titles[-20:])}

Please rank the following {len(candidate_items)} items by likelihood of purchase.

Candidate items:
{candidates_str}

IMPORTANT: Your response must end with exactly one line in this format:
RANKING: number1,number2,number3,number4,number5,number6,number7,number8,number9,number10,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20

Where each number is between 1-{len(candidate_items)}. Example:
RANKING: 26,45,78,50,38,99,77,43,53,89,8,93,97,52,47,31,48,83,98,79
"""
            
            # 调用API
            response, reasoning, raw_response = self.call_api(prompt)
            
            # 解析响应获得排序
            predicted_ranking = self.parse_ranking_response(response, candidate_items)
            
            # 更精确的解析成功率检查
            # parsing_success 现在表示是否成功解析出了推荐列表（不要求完整）
            parsing_success = (len(predicted_ranking) > 0)
            
            # 计算目标商品在预测中的排名（1-indexed for display）
            if target_id in predicted_ranking:
                target_rank_in_prediction = predicted_ranking.index(target_id) + 1
            else:
                # 目标不在预测的top-K中，设为预测长度+1（表示排在预测列表外）
                target_rank_in_prediction = len(predicted_ranking) + 1
            
            # 计算指标
            sample_metrics = self.calculate_metrics_like_training(predicted_ranking, target_id)
            
            # 累积指标
            for key in accumulated_metrics:
                accumulated_metrics[key] += sample_metrics[key]
            
            valid_samples += 1
            
            # 保存详细结果
            if save_results:
                detailed_results.append({
                    'sample_id': sample_index,
                    'user_id': user_id,
                    'history_titles': history_titles[-10:],
                    'target_title': target_title,
                    'target_rank': target_rank_in_prediction,
                    'predicted_items_count': len(predicted_ranking),  # 新增：实际预测的商品数量
                    'total_candidates': len(candidate_items),
                    'parsing_success': parsing_success,
                    'target_candidate_position': target_candidate_position,
                    'metrics': sample_metrics,
                    'api_response': response[:1000],  # 截断保存
                    'reasoning_preview': reasoning[:1000] if reasoning else None  # 新增：思考过程预览
                })
                
                # 保存详细的prompt和response用于调试
                prompt_response_logs.append({
                    'sample_id': sample_index,
                    'user_id': user_id,
                    'prompt': prompt,
                    'response': response,
                    'reasoning': reasoning,  # 新增：完整的思考过程
                    'raw_response': raw_response,  # 新增：原始响应用于调试
                    'candidate_items': candidate_items,
                    'predicted_ranking': predicted_ranking,
                    'target_id': target_id,
                    'target_rank': target_rank_in_prediction,
                    'target_candidate_position': target_candidate_position,
                    'parsing_success': parsing_success,
                    'sample_metrics': sample_metrics
                })
            
            # 实时进度显示：每progress_interval个样本输出一次中间结果
            if valid_samples % progress_interval == 0:
                current_metrics = {k: v / valid_samples for k, v in accumulated_metrics.items()}
                print(f"\n--- Progress Update: {valid_samples}/{total_to_evaluate} samples ---")
                print(f"Current NDCG@10: {current_metrics['ndcg@10']:.6f}")
                print(f"Current Hit Rate@10: {current_metrics['hit_rate@10']:.6f}")
                print(f"Target found in top-10: {sum(1 for r in detailed_results[-progress_interval:] if r['target_rank'] <= 10)}/{min(progress_interval, len(detailed_results))} samples")
                print(f"Parsing success rate: {sum(1 for r in detailed_results[-progress_interval:] if r['parsing_success'])}/{min(progress_interval, len(detailed_results))} samples")
                print("-" * 50)
            
            # 增量保存：每save_interval个样本保存一次中间结果
            if save_results and valid_samples % save_interval == 0:
                self._save_incremental_results(
                    accumulated_metrics, detailed_results, prompt_response_logs,
                    valid_samples, split, total_to_evaluate, progress_interval,
                    preserve_order
                )
        
        # 计算平均指标，与训练脚本的MetricUpdater.compute()一致
        if valid_samples > 0:
            for key in accumulated_metrics:
                accumulated_metrics[key] = accumulated_metrics[key] / valid_samples
        
        accumulated_metrics['total_samples'] = valid_samples
        
        # 打印最终结果，格式与训练脚本一致
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION RESULTS ({valid_samples} samples)")
        print(f"{'='*60}")
        print(f"hit_rate@5:  {accumulated_metrics['hit_rate@5']:.6f}")
        print(f"hit_rate@10: {accumulated_metrics['hit_rate@10']:.6f}")
        print(f"hit_rate@20: {accumulated_metrics['hit_rate@20']:.6f}")
        print(f"ndcg@5:      {accumulated_metrics['ndcg@5']:.6f}")
        print(f"ndcg@10:     {accumulated_metrics['ndcg@10']:.6f}")  # 与训练日志格式一致
        print(f"ndcg@20:     {accumulated_metrics['ndcg@20']:.6f}")
        
        # 计算和显示额外的分析指标
        if detailed_results:
            target_in_top10_count = sum(1 for r in detailed_results if r['target_rank'] <= 10)
            parsing_success_count = sum(1 for r in detailed_results if r['parsing_success'])
            avg_predicted_items = sum(r['predicted_items_count'] for r in detailed_results) / len(detailed_results)
            print(f"\nAdditional Analysis:")
            print(f"Target in Top-10: {target_in_top10_count}/{len(detailed_results)} ({target_in_top10_count/len(detailed_results)*100:.1f}%)")
            print(f"Parsing Success: {parsing_success_count}/{len(detailed_results)} ({parsing_success_count/len(detailed_results)*100:.1f}%)")
            print(f"Average Predicted Items: {avg_predicted_items:.1f} (vs {candidate_count} candidates)")
            target_positions = [r['target_candidate_position'] for r in detailed_results if r.get('target_candidate_position')]
            if target_positions:
                avg_target_position = sum(target_positions) / len(target_positions)
                print(f"Average Target Position in Candidates: {avg_target_position:.1f} ({self.target_position_mode})")
            
            # 显示一些示例
            print(f"\nExample predictions (first 3 samples):")
            for i, result in enumerate(detailed_results[:3]):
                print(f"  Sample {result['sample_id']}: Target rank {result['target_rank']}/{result['predicted_items_count']} predicted, NDCG@10={result['metrics']['ndcg@10']:.4f}")
        
        # 保存结果
        if save_results:
            # 保存主要结果
            results_file = os.path.join(
                self.output_dir,
                f"deepseek_api_results_{split}_{self.model_name.replace('/', '_')}.json"
            )
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metrics': accumulated_metrics,
                    'detailed_results': detailed_results,
                    'config': {
                        'dataset_dir': self.dataset_dir,
                        'model_name': self.model_name,
                        'num_samples': total_to_evaluate,
                        'split': split,
                        'evaluation_method': 'training_consistent',
                        'api_base': self.api_base,
                        'progress_interval': progress_interval,
                        'seed': self.seed,
                        'target_position_mode': self.target_position_mode,
                        'output_dir': self.output_dir,
                        'test_sample_limit': self.test_limit,
                        'preserve_order': preserve_order
                    }
                }, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed results saved to {results_file}")
            
            # 保存prompt和response日志用于调试
            if prompt_response_logs:
                debug_log_file = os.path.join(
                    self.output_dir,
                    f"api_debug_logs_{split}_{self.model_name.replace('/', '_')}.json"
                )
                with open(debug_log_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'logs': prompt_response_logs,
                        'summary': {
                            'total_samples': len(prompt_response_logs),
                            'parsing_success_rate': sum(1 for log in prompt_response_logs if log['parsing_success']) / len(prompt_response_logs),
                            'avg_target_rank': sum(log['target_rank'] for log in prompt_response_logs) / len(prompt_response_logs),
                            'config': {
                                'dataset_dir': self.dataset_dir,
                                'model_name': self.model_name,
                                'split': split,
                                'target_position_mode': self.target_position_mode,
                                'output_dir': self.output_dir,
                                'test_sample_limit': self.test_limit,
                                'preserve_order': preserve_order
                            }
                        }
                    }, f, indent=2, ensure_ascii=False)
                print(f"Debug logs (prompts & responses) saved to {debug_log_file}")
        
        return accumulated_metrics
    
    def _save_incremental_results(self, accumulated_metrics, detailed_results, prompt_response_logs,
                                 valid_samples, split, num_samples, progress_interval,
                                 preserve_order: bool = False):
        """增量保存中间结果，防止程序中断导致数据丢失"""
        try:
            # 计算当前平均指标
            current_metrics = {k: v / valid_samples for k, v in accumulated_metrics.items()}
            current_metrics['total_samples'] = valid_samples
            
            # 保存中间结果文件（带时间戳）
            import time
            timestamp = int(time.time())
            
            # 主要结果的增量保存
            incremental_file = os.path.join(
                self.output_dir,
                f"api_incremental_{split}_{self.model_name.replace('/', '_')}_{valid_samples}samples_{timestamp}.json"
            )
            with open(incremental_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'current_metrics': current_metrics,
                    'detailed_results': detailed_results,
                    'status': {
                        'completed_samples': valid_samples,
                        'total_samples': num_samples,
                        'progress_percentage': (valid_samples / num_samples * 100) if num_samples else 0,
                        'timestamp': timestamp,
                        'is_incremental': True
                    },
                    'config': {
                        'dataset_dir': self.dataset_dir,
                        'model_name': self.model_name,
                        'split': split,
                        'evaluation_method': 'training_consistent',
                        'api_base': self.api_base,
                        'progress_interval': progress_interval,
                        'seed': self.seed,
                        'target_position_mode': self.target_position_mode,
                        'output_dir': self.output_dir,
                        'test_sample_limit': self.test_limit,
                        'preserve_order': preserve_order
                    }
                }, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Incremental save: {incremental_file} ({valid_samples} samples)")
            
            # 保存最新的调试日志（覆盖模式）
            if prompt_response_logs:
                debug_incremental_file = os.path.join(
                    self.output_dir,
                    f"api_debug_incremental_{split}_{self.model_name.replace('/', '_')}.json"
                )
                with open(debug_incremental_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'logs': prompt_response_logs,
                        'summary': {
                            'total_samples': len(prompt_response_logs),
                            'parsing_success_rate': sum(1 for log in prompt_response_logs if log['parsing_success']) / len(prompt_response_logs),
                            'avg_target_rank': sum(log['target_rank'] for log in prompt_response_logs) / len(prompt_response_logs),
                            'timestamp': timestamp,
                            'is_incremental': True,
                            'target_position_mode': self.target_position_mode,
                            'output_dir': self.output_dir,
                            'test_sample_limit': self.test_limit,
                            'preserve_order': preserve_order
                        }
                    }, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"⚠️  Warning: Failed to save incremental results: {e}")
    
    def run_full_evaluation(self, 
                          test_samples: Optional[int] = None,  # 改为可选，默认全测试集
                          valid_samples: Optional[int] = None,  # 改为可选，默认全验证集
                          progress_interval: int = 100,  # 添加进度间隔参数
                          save_interval: int = 50,  # 新增：保存间隔参数
                          test_only: bool = False):  # 新增：仅测试测试集
        """运行完整评估，在验证集和测试集上都进行测试，与训练脚本一致"""
        print("=" * 60)
        print(f"API Recommendation System Evaluation (Training Consistent)")
        print(f"Dataset: {self.dataset_dir}")
        print(f"Model: {self.model_name}")
        print(f"API Base: {self.api_base}")
        print(f"Evaluation Method: Full item library ranking (like training)")
        print(f"Progress interval: Every {progress_interval} samples")
        print(f"Save interval: Every {save_interval} samples")
        print(f"Target position mode: {self.target_position_mode}")
        print(f"Output directory: {self.output_dir}")
        if self.test_limit is not None:
            print(f"Test sample limit (first-N in order when applicable): {self.test_limit}")
        if test_only:
            print(f"💰 Cost-saving mode: Testing ONLY the test set")
        print("=" * 60)
        
        self.test_only_mode = test_only
        preserve_order_for_test = test_only and self.test_limit is not None

        valid_metrics = None
        
        # 验证集评估（仅在非test_only模式下）
        if not test_only:
            print("\n1. Validation Set Evaluation")
            valid_metrics = self.evaluate_like_training(
                num_samples=valid_samples,  # None表示全集
                split='valid',
                progress_interval=progress_interval,  # 使用传入的进度间隔
                save_interval=save_interval,  # 使用传入的保存间隔
                preserve_order=False
            )
        else:
            print("\n⏭️  Skipping validation set evaluation (test_only mode)")
        
        # 测试集评估
        eval_number = "1" if test_only else "2"
        print(f"\n{eval_number}. Test Set Evaluation")
        requested_test_samples = self.test_limit if preserve_order_for_test else test_samples
        test_metrics = self.evaluate_like_training(
            num_samples=requested_test_samples,  # None表示全集
            split='test',
            progress_interval=progress_interval,  # 使用传入的进度间隔
            save_interval=save_interval,  # 使用传入的保存间隔
            preserve_order=preserve_order_for_test
        )
        
        # 汇总结果
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY (Training Consistent)")
        print("=" * 60)
        
        if valid_metrics:
            print("\nValidation Set:")
            for metric, value in valid_metrics.items():
                if metric != 'total_samples':
                    print(f"  {metric}: {value:.4f}")
        
        print("\nTest Set:")
        for metric, value in test_metrics.items():
            if metric != 'total_samples':
                print(f"  {metric}: {value:.4f}")
        
        # 与训练日志的目标指标对比
        train_target_ndcg = 0.012233516966979588
        test_ndcg = test_metrics.get('ndcg@10', 0)
        print(f"\n📊 Comparison with training target:")
        print(f"  Training model ndcg@10: {train_target_ndcg:.6f}")
        print(f"  {self.model_name} ndcg@10:  {test_ndcg:.6f}")
        print(f"  Difference: {test_ndcg - train_target_ndcg:+.6f}")
        
        # 保存汇总结果
        summary_results = {
            'test': test_metrics,
            'config': {
                'dataset_dir': self.dataset_dir,
                'model_name': self.model_name,
                'api_base': self.api_base,
                'test_samples_requested': requested_test_samples,
                'test_samples_evaluated': test_metrics.get('total_samples'),
                'evaluation_method': 'training_consistent',
                'test_only_mode': test_only,
                'seed': self.seed,
                'target_position_mode': self.target_position_mode,
                'output_dir': self.output_dir,
                'test_sample_limit': self.test_limit,
                'preserve_order_for_test': preserve_order_for_test
            }
        }
        
        if valid_metrics:
            summary_results['validation'] = valid_metrics
            summary_results['config']['valid_samples_requested'] = valid_samples
            summary_results['config']['valid_samples_evaluated'] = valid_metrics.get('total_samples')
        
        summary_file = os.path.join(
            self.output_dir,
            f"api_consistent_summary_{self.model_name.replace('/', '_')}.json"
        )
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        print(f"\nSummary results saved to {summary_file}")
        
        if test_only:
            return test_metrics
        else:
            return valid_metrics, test_metrics

def main():
    parser = argparse.ArgumentParser(description="Test API with training-consistent evaluation")
    parser.add_argument("--api_key", type=str, default="sk-ZUK05LeE0U3FVQAp6e5eCd3628774c4181D513786bD3B901", help="API key")
    parser.add_argument("--dataset_dir", type=str, default="/data/hongdeyao/Movies_and_TV_0_2022-10-2023-10", help="Dataset directory")
    parser.add_argument("--test_samples", type=int, default=200, help="Number of test samples (None for full dataset)")
    parser.add_argument("--test_limit", type=int, default=None,
                        help="In test-only mode, evaluate only the first N samples from the test split")
    parser.add_argument("--valid_samples", type=int, default=None, help="Number of validation samples (None for full dataset)")
    parser.add_argument("--api_base", type=str, default="http://115.182.62.174:18888/v1", help="API base URL")
    parser.add_argument("--model_name", type=str, default="deepseek/deepseek-r1", 
                       choices=["deepseek/deepseek-r1"],
                       help="Model name")
    parser.add_argument("--sample_for_api_efficiency", action="store_true", 
                       help="Use sampling for API efficiency (default: False, test full dataset like training)")
    parser.add_argument("--progress_interval", type=int, default=2, 
                       help="Show progress every N samples (default: 50)")
    parser.add_argument("--save_interval", type=int, default=5, 
                       help="Save incremental results every N samples (default: 25)")
    parser.add_argument("--test_only", action="store_true", 
                       help="Only evaluate test set (saves API costs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for candidate sampling")
    parser.add_argument("--target_position", type=str, default="random",
                        choices=["first", "last", "random"],
                        help="Control where the correct item appears in candidates (default: first)")
    parser.add_argument("--output_dir", type=str, default="/data/hongdeyao/code/RRec/sft/outputs/movies_200_sample",
                        help="Directory to save evaluation outputs (default: current directory)")
    parser.add_argument("--candidate_mode", type=str, default="json",
                        choices=["random", "json"],
                        help="Candidate selection mode: random sampling or predefined JSON")
    parser.add_argument("--candidate_json_path", type=str, default="/data/hongdeyao/code/RRec/sft/outputs/movies_200_sample/checkpoint-009000.json",
                        help="Path to predefined candidate JSON file when candidate_mode=json")
    parser.add_argument("--candidate_json_split", type=str, default="test",
                        choices=["train", "valid", "test"],
                        help="Dataset split to pair with the predefined candidate JSON entries")
    
    args = parser.parse_args()
    
    # 设置随机种子以确保可复现性
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 如果启用采样模式，设置默认样本数
    if args.sample_for_api_efficiency:
        test_samples = args.test_samples or 1000
        valid_samples = args.valid_samples or 500
        print("⚠️  API efficiency mode enabled: using sampling instead of full dataset")
    else:
        test_samples = args.test_samples  # None表示全集
        valid_samples = args.valid_samples  # None表示全集
        print("✅ Full dataset evaluation mode (same as training script)")

    if args.test_limit is not None and not args.test_only:
        print("ℹ️  test_limit is set but test_only mode is disabled; limit will apply only when test_only is True.")

    if args.candidate_mode == "json":
        candidate_json_path = os.path.abspath(args.candidate_json_path)
        print(f"📄 Candidate selection mode: JSON (path={candidate_json_path})")
    else:
        candidate_json_path = None
        print("🎲 Candidate selection mode: random sampling from dataset")
    
    # 初始化测试器
    tester = APIRecommendationTester(
        api_key=args.api_key,
        dataset_dir=args.dataset_dir,
        api_base=args.api_base,
        model_name=args.model_name,
        seed=args.seed,
        target_position=args.target_position,
        output_dir=args.output_dir,
        test_limit=args.test_limit,
        predefined_candidates_path=candidate_json_path,
        predefined_candidate_split=args.candidate_json_split
    )
    
    # 运行完整评估
    tester.run_full_evaluation(
        test_samples=test_samples,
        valid_samples=valid_samples,
        progress_interval=args.progress_interval,
        save_interval=args.save_interval,
        test_only=args.test_only
    )

if __name__ == "__main__":
    main()
