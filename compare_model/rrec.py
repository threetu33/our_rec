#!/usr/bin/env python3
"""
完全修复版本的R1兼容测试脚本
严格按照训练时的评估流程进行推理
"""

import os
import sys
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from datasets import load_from_disk
import argparse
from datetime import datetime

# 添加项目路径
sys.path.append('/data2/home/hongdeyao/RRec')

from paths import model_names
from trainers.utils import get_tokenizer, calculate_metrics, Similarity
from peft import PeftModel
from prompters.rrec_prompter import UserGenPrompter, UserPrompter
from prompters.prompts import obtain_prompts

class TrainingConsistentTester:
    def __init__(self, 
                 checkpoint_path: str,
                 dataset_dir: str,
                 model_type: str = "qwen",
                 device: str = "cuda:6",
                 reference_json_path: Optional[str] = None,
                 user_window_size: int = 10):
        """
        初始化训练一致性测试器
        
        Args:
            checkpoint_path: checkpoint路径
            dataset_dir: 数据集目录
            model_type: 模型类型
            device: 使用的设备
            reference_json_path: 参考JSON文件路径,包含固定的user_id和candidate_ids
        """
        self.checkpoint_path = checkpoint_path
        self.dataset_dir = dataset_dir
        self.model_type = model_type
        self.device = device
        self.reference_json_path = reference_json_path
        self.reference_data = None
        self.user_window_size = max(1, int(user_window_size))
        
        print(f"🚀 Initializing Training-Consistent Tester")
        print(f"📁 Checkpoint: {checkpoint_path}")
        print(f"📊 Dataset: {dataset_dir}")
        print(f"🤖 Model: {model_type}")
        print(f"💻 Device: {device}")
        
        # 加载数据集
        self.load_dataset()
        
        # 加载模型
        self.load_model()
        
        # 初始化prompters（与训练时完全一致）
        self.init_prompters()
        
        # 初始化相似度计算器
        self.init_similarity()
        
        # 加载参考JSON数据(如果提供)
        if self.reference_json_path:
            self.load_reference_json()
        
    def load_reference_json(self):
        """加载参考JSON文件,包含固定的测试数据"""
        print(f"📂 Loading reference JSON from {self.reference_json_path}")
        try:
            with open(self.reference_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.reference_data = data.get('detailed_results', [])
            
            if self.reference_data:
                print(f"✅ Loaded {len(self.reference_data)} reference samples from JSON")
                print(f"  Sample fields: {list(self.reference_data[0].keys())}")
            else:
                print("⚠️  No detailed_results found in JSON file")
                self.reference_data = None
                
        except Exception as e:
            print(f"❌ Error loading reference JSON: {e}")
            self.reference_data = None
    
    def load_dataset(self):
        """加载数据集"""
        print(f"📂 Loading dataset from {self.dataset_dir}")
        try:
            self.dataset = load_from_disk(self.dataset_dir)
            self.test_data = self.dataset['test']
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
        """创建商品映射"""
        self.id2title = {}
        self.all_item_ids = []
        
        # 创建user_id到样本索引的映射
        # 注意：同一 user_id 可能在 test split 中出现多次，这里同时维护：
        # - user_id_to_sample：首次出现的样本索引（向后兼容，且避免被覆盖）
        # - user_id_to_samples：该 user_id 对应的所有样本索引列表
        self.user_id_to_sample = {}
        self.user_id_to_samples = {}
        
        for item in self.item_info:
            item_id = item['item_id']
            title = item.get('title', f'Unknown Item {item_id}')
            
            self.id2title[item_id] = title
            
            if item_id != 0:  # 排除padding token
                self.all_item_ids.append(item_id)
        
        # 创建user_id索引
        for idx, sample in enumerate(self.test_data):
            user_id = sample.get('user_id', None)
            if user_id:
                # 仅当首次遇到该 user_id 时记录，避免被后续相同 user 覆盖
                if user_id not in self.user_id_to_sample:
                    self.user_id_to_sample[user_id] = idx
                # 记录所有位置，便于调试或需要精确定位时使用
                self.user_id_to_samples.setdefault(user_id, []).append(idx)
        
        print(f"📋 Created mappings for {len(self.id2title)} items (including {len(self.all_item_ids)} non-pad items)")
        print(f"📋 Created user_id mappings for {len(self.user_id_to_sample)} distinct users")
    
    def load_model(self):
        """加载模型"""
        print(f"🔄 Loading model from {self.checkpoint_path}")
        
        try:
            # 根据模型类型选择对应的类
            if self.model_type == 'qwen':
                model_name = model_names["Qwen2.5-3B-Instruct"]
                from models.qwen_models import (Qwen2RRecCasualLM as ModelClass,
                                               Qwen2RRecConfig as ConfigClass)
            elif self.model_type == 'gemma':
                model_name = model_names["Gemma-2-2b-it"]
                from models.gemma_models import (Gemma2RRecCasualLM as ModelClass,
                                                Gemma2RRecConfig as ConfigClass)
            else:
                raise NotImplementedError(f"Model type {self.model_type} not supported")
            
            # 加载tokenizer
            self.tokenizer = get_tokenizer(model_name)
            
            # 加载配置
            config = ConfigClass.from_pretrained(model_name)
            config.use_cache = False
            config.pad_token_id = self.tokenizer.pad_token_id
            
            # 加载基础模型
            print(f"📥 Loading base model: {model_name}")
            self.model = ModelClass.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": self.device},
                config=config
            )
            
            # 加载LoRA权重
            print(f"🔧 Loading LoRA weights from {self.checkpoint_path}")
            self.model = PeftModel.from_pretrained(self.model, self.checkpoint_path)
            
            # 合并LoRA权重以提高推理速度
            print("🔄 Merging LoRA weights...")
            self.model = self.model.merge_and_unload()
            
            self.model.eval()
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def init_prompters(self):
        """初始化prompters，与训练时完全一致"""
        print("🔄 Initializing prompters...")
        
        # 从数据集目录提取category
        dataset_name = self.dataset_dir.split('/')[-1]
        if 'Musical_Instruments' in dataset_name:
            category = 'Musical_Instruments'
        elif 'Video_Games' in dataset_name:
            category = 'Video_Games'
        elif 'CDs_and_Vinyl' in dataset_name:
            category = 'CDs_and_Vinyl'
        else:
            # 默认使用Musical_Instruments
            category = 'Musical_Instruments'
            print(f"⚠️  Could not determine category from {dataset_name}, using default: {category}")
        
        # 设置与训练时相同的参数（可配置的 user window size）
        window_size = self.user_window_size
        emb_token = '<answer>'
        emb_end_token = '</answer>'
        
        # 创建完整的数据集结构（模拟训练时的full_dataset）
        full_dataset = {
            'test': self.test_data,
            'item_info': self.item_info
        }
        
        # 初始化UserGenPrompter（用于生成user reasoning的prompt）
        self.user_gen_prompter = UserGenPrompter(
            dset=full_dataset,
            tokenizer=self.tokenizer,
            category=category,
            window_size=window_size,
            emb_token=emb_token,
            emb_end_token=emb_end_token
        )
        
        # 初始化UserPrompter（用于将reasoning转换为最终输入）
        self.user_prompter = UserPrompter(
            dset=full_dataset,
            tokenizer=self.tokenizer,
            category=category,
            window_size=window_size,
            input_ids_max_length=2048,
            emb_token=emb_token,
            emb_end_token=emb_end_token
        )
        
        print("✅ Prompters initialized")
    
    def init_similarity(self):
        """初始化相似度计算器"""
        # 创建与训练时相同的相似度配置
        class SimilarityConfig:
            similarity_type = "dot"
            similarity_temperature = 0.02
            similarity_normalization = True
        
        config = SimilarityConfig()
        self.similarity = Similarity(config)
        print("✅ Similarity calculator initialized")
    
    def generate_item_embeddings(self):
        """生成所有商品的embeddings，与训练时完全一致"""
        print("🔄 Generating item embeddings...")
        
        self.item_embeddings = {}
        batch_size = 32
        
        # 准备所有商品的输入，使用与训练时相同的格式
        all_items = []
        for item in self.item_info:
            if item['item_id'] != 0:  # 排除padding
                # 使用与训练时相同的item prompt格式
                title = item.get('title', f"Unknown Item {item['item_id']}")
                description = item.get('description', '')
                if isinstance(description, list):
                    description = ' '.join(description[::-1]) if description else ''
                
                # 构建item info string（与prompter中的格式一致）
                average_rating = item.get('average_rating', 0.0)
                num_buyers = item.get('rating_number', 0)
                
                item_info_str = (f"Title: {title}\n"
                               f"User Rating: {average_rating}\n"
                               f"Number of Buyers: {num_buyers}\n"
                               f"Description: {description}")
                
                # 获取category（与init_prompters中的逻辑一致）
                dataset_name = self.dataset_dir.split('/')[-1]
                if 'Musical_Instruments' in dataset_name:
                    category = 'Musical_Instruments'
                elif 'Video_Games' in dataset_name:
                    category = 'Video_Games'
                elif 'CDs_and_Vinyl' in dataset_name:
                    category = 'CDs_and_Vinyl'
                else:
                    category = 'Musical_Instruments'
                
                prompts = obtain_prompts(category)
                item_prompt = prompts['item_prompt'].format(
                    emb_token='<answer>',
                    emb_end_token='</answer>'
                )
                
                # 构建完整的item prompt
                full_prompt = item_prompt + '\n\n' + item_info_str
                
                # if "Milisten 2PCS Humbucker Pickup Ring Frame Mounting Ring For Electric Guitars GB301 Golden" in full_prompt:
                #     print(full_prompt)
                
                # 使用chat template
                conversation = [
                    {"role": "user", "content": full_prompt},
                    {"role": "assistant", "content": "<answer>"}
                ]
                
                item_text = self.tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                
                all_items.append((item['item_id'], item_text))
        
        # 批量处理
        with torch.no_grad():
            for i in tqdm(range(0, len(all_items), batch_size), desc="Generating item embeddings"):
                batch_items = all_items[i:i+batch_size]
                item_ids = [item[0] for item in batch_items]
                item_texts = [item[1] for item in batch_items]
                
                # Tokenize
                inputs = self.tokenizer(
                    item_texts,
                    padding=True,
                    truncation=True,
                    max_length=768,  # 与训练时的item_input_max_length一致
                    return_tensors="pt"
                ).to(self.device)
                
                # 获取embeddings
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    return_causal_output=False,
                    return_with_last_hidden_states=True
                )
                
                # 保存embeddings
                for j, item_id in enumerate(item_ids):
                    self.item_embeddings[item_id] = outputs[j].cpu()
        
        print(f"✅ Generated embeddings for {len(self.item_embeddings)} items")
    
    def generate_user_reasoning_training_style(self, sample: Dict) -> str:
        """使用训练时的方式生成user reasoning"""
        # 重命名列以匹配训练时的格式
        sample_copy = dict(sample)
        sample_copy['seq_input_ids'] = sample_copy['history_item_id']
        sample_copy['seq_labels'] = sample_copy['item_id']
        
        # 使用UserGenPrompter生成prompt
        chat_example = self.user_gen_prompter.to_chat_example(sample_copy)
        # 保存生成的prompt内容，便于后续存储
        self.last_user_reasoning_prompt = chat_example.get('prompt', None)
        
        # 转换为tensor格式
        tensor_example = self.user_gen_prompter.totensor(
            chat_example, 
            max_length=2048
        )
        
        # 准备输入
        input_ids = torch.tensor(tensor_example['input_ids']).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(tensor_example['attention_mask']).unsqueeze(0).to(self.device)
        
        # 生成reasoning
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=300,  # 与训练时一致
                temperature=1.5,
                top_k=200,
                top_p=1.0,
                do_sample=True,
                # temperature=0.0,
                # do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stop_strings=['<answer>'],  # 与训练时的stop token一致
                tokenizer=self.tokenizer,  # 添加tokenizer参数
            )
        
        # 解码生成的reasoning
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # 确保以<answer>结尾（与训练时一致）
        if not generated_text.endswith('<answer>'):
            generated_text += '<answer>'
        
        return generated_text
    
    def get_user_embedding_training_style(self, sample: Dict, reasoning: str) -> torch.Tensor:
        """使用训练时的方式从reasoning提取user embedding"""
        # 创建包含reasoning的样本
        sample_with_reasoning = dict(sample)
        sample_with_reasoning['seq_input_ids'] = sample_with_reasoning['history_item_id']
        sample_with_reasoning['seq_labels'] = sample_with_reasoning['item_id']
        sample_with_reasoning['profile'] = reasoning
        
        # 使用UserPrompter转换（与训练时的evaluate方法一致）
        chat_example = self.user_prompter.to_chat_example(sample_with_reasoning)
        
        # 转换为tensor格式
        tensor_example = self.user_prompter.totensor(
            chat_example,
            max_length=2048,
            continue_final_message=True
        )
        
        # 准备输入
        input_ids = torch.tensor(tensor_example['input_ids']).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(tensor_example['attention_mask']).unsqueeze(0).to(self.device)
        
        # 获取最后一个token的embedding（与训练时一致）
        with torch.no_grad():
            user_embedding = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_causal_output=False,
                return_with_last_hidden_states=True
            )
        
        return user_embedding.squeeze(0)  # 移除batch维度
    
    def compute_similarity_training_style(self, user_embedding: torch.Tensor, candidate_items: List[int]) -> torch.Tensor:
        """使用训练时的方式计算相似度"""
        # 获取候选商品的embeddings
        candidate_embeddings = []
        for item_id in candidate_items:
            if item_id in self.item_embeddings:
                candidate_embeddings.append(self.item_embeddings[item_id])
            else:
                # 如果没有embedding，使用零向量
                candidate_embeddings.append(torch.zeros_like(user_embedding))
        
        candidate_embeddings = torch.stack(candidate_embeddings).to(self.device)
        user_embedding = user_embedding.to(self.device)
        
        # 使用训练时的相似度计算方式
        similarity_scores = self.similarity(user_embedding.unsqueeze(0), candidate_embeddings)
        
        return similarity_scores.squeeze(0)  # 移除batch维度
    
    def calculate_metrics_like_training(self, predicted_ranking: List[int], target_id: int) -> Dict[str, float]:
        """
        计算评估指标，与训练脚本完全一致
        """
        metrics = {}
        
        # 找到目标商品在预测排序中的位置（0-indexed）
        try:
            target_position = predicted_ranking.index(target_id)
        except ValueError:
            # 目标商品不在预测中，设为列表长度（表示排在最后）
            target_position = len(predicted_ranking)
        
        # 计算Hit Rate @ K
        for k in [5, 10, 20]:
            metrics[f'hit_rate@{k}'] = 1.0 if target_position < k else 0.0
        
        # 计算DCG @ K（与训练脚本完全一致）
        for k in [5, 10, 20]:
            if target_position < k:
                # 与训练脚本一致：position 0对应log2(2), position 1对应log2(3)...
                dcg_value = 1.0 / np.log2(target_position + 2)
                metrics[f'ndcg@{k}'] = dcg_value
            else:
                metrics[f'ndcg@{k}'] = 0.0
        
        return metrics
    
    def evaluate_training_consistent(self, num_samples: int = 622, save_results: bool = True) -> Dict[str, float]:
        """
        训练一致性评估函数
        
        Args:
            num_samples: 测试样本数量
            save_results: 是否保存详细结果
            
        Returns:
            评估指标字典
        """
        # 如果提供了参考JSON,使用固定数据评估
        if self.reference_data is not None:
            return self.evaluate_with_reference_data(save_results=save_results)
        
        print(f"\n🎯 Starting Training-Consistent evaluation on first {num_samples} test samples")
        print("=" * 60)
        
        # 生成商品embeddings
        self.generate_item_embeddings()
        
        # 初始化累积指标
        accumulated_metrics = {
            'hit_rate@5': 0.0, 'hit_rate@10': 0.0, 'hit_rate@20': 0.0,
            'ndcg@5': 0.0, 'ndcg@10': 0.0, 'ndcg@20': 0.0
        }
        
        valid_samples = 0
        detailed_results = []
        
        for i in tqdm(range(min(num_samples, len(self.test_data))), desc="Evaluating"):
            sample = self.test_data[i]
            
            # 获取历史和目标
            history_ids = sample['history_item_id']
            target_id = sample['item_id']
            
            # 跳过padding token
            if target_id == 0:
                continue
                
            # 过滤历史中的padding token
            history_ids = [id for id in history_ids if id != 0]
            history_titles = [self.id2title.get(id, f"Unknown_{id}") for id in history_ids]
            target_title = self.id2title.get(target_id, f"Unknown_{target_id}")
            
            # 构建可用商品集合：排除历史交互商品
            available_items = [item_id for item_id in self.all_item_ids 
                             if item_id not in history_ids]
            
            # 确保目标商品在可用商品中
            if target_id not in available_items:
                print(f"Warning: Target item {target_id} is in history for sample {i}")
                continue
            
            # R1采样逻辑：随机选择20个候选商品
            sample_size = min(20, len(available_items))
            
            # 确保目标商品在候选中
            other_items = [item for item in available_items if item != target_id]
            if len(other_items) >= sample_size - 1:
                sampled_others = random.sample(other_items, sample_size - 1)
                candidate_items = [target_id] + sampled_others
            else:
                candidate_items = available_items
            
            random.shuffle(candidate_items)
            
            # 生成用户推理（与训练时完全一致）
            user_reasoning = self.generate_user_reasoning_training_style(sample)
            
            # 从推理中提取用户embedding（与训练时完全一致）
            user_embedding = self.get_user_embedding_training_style(sample, user_reasoning)
            
            # 计算相似度（与训练时完全一致）
            similarity_scores = self.compute_similarity_training_style(user_embedding, candidate_items)
            
            # 排序获得预测结果
            _, sorted_indices = torch.sort(similarity_scores, descending=True)
            predicted_ranking = [candidate_items[idx] for idx in sorted_indices.cpu().tolist()]
            
            # 计算指标
            sample_metrics = self.calculate_metrics_like_training(predicted_ranking, target_id)
            
            # 累积指标
            for key in accumulated_metrics:
                accumulated_metrics[key] += sample_metrics[key]
            
            valid_samples += 1
            
            # 保存详细结果
            if save_results:
                target_rank = predicted_ranking.index(target_id) + 1 if target_id in predicted_ranking else len(predicted_ranking) + 1
                detailed_results.append({
                    'sample_id': i,
                    'user_id': sample.get('user_id', 'unknown'),
                    'history_titles': history_titles[-self.user_window_size:],  # 保存最后N条历史
                    'target_title': target_title,
                    'target_id': target_id,
                    'target_rank': target_rank,
                    'total_candidates': len(candidate_items),
                    'candidate_items': candidate_items,  # 保存候选商品列表，方便与R1对比
                    'predicted_ranking': predicted_ranking,  # 新增字段，顺序即为排名
                    'user_reasoning': user_reasoning,  # 保存完整的reasoning，不截断
                    'user_reasoning_prompt': getattr(self, 'last_user_reasoning_prompt', None),  # 新增，保存生成reasoning时的prompt
                    'metrics': sample_metrics
                })

            # 每1个样本输出一次进度
            if valid_samples % 1 == 0:
                current_metrics = {k: v / valid_samples for k, v in accumulated_metrics.items()}
                print(f"\n--- Progress: {valid_samples}/{num_samples} samples ---")
                print(f"Current NDCG@10: {current_metrics['ndcg@10']:.6f}")
                print(f"Current Hit Rate@10: {current_metrics['hit_rate@10']:.6f}")
        
        # 计算平均指标
        if valid_samples > 0:
            for key in accumulated_metrics:
                accumulated_metrics[key] = accumulated_metrics[key] / valid_samples
        
        accumulated_metrics['total_samples'] = valid_samples
        
        # 打印最终结果
        print(f"\n{'='*60}")
        print(f"FINAL TRAINING-CONSISTENT EVALUATION RESULTS ({valid_samples} samples)")
        print(f"{'='*60}")
        print(f"hit_rate@5:  {accumulated_metrics['hit_rate@5']:.6f}")
        print(f"hit_rate@10: {accumulated_metrics['hit_rate@10']:.6f}")
        print(f"hit_rate@20: {accumulated_metrics['hit_rate@20']:.6f}")
        print(f"ndcg@5:      {accumulated_metrics['ndcg@5']:.6f}")
        print(f"ndcg@10:     {accumulated_metrics['ndcg@10']:.6f}")
        print(f"ndcg@20:     {accumulated_metrics['ndcg@20']:.6f}")
        
        # 保存结果
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f"training_consistent_results_{self.model_type}_{num_samples}samples_{timestamp}.json"
            
            results = {
                'metrics': accumulated_metrics,
                'detailed_results': detailed_results,
                'config': {
                    'checkpoint_path': self.checkpoint_path,
                    'dataset_dir': self.dataset_dir,
                    'model_type': self.model_type,
                    'num_samples': num_samples,
                    'evaluation_method': 'training_consistent',
                    'device': self.device,
                    'note': 'This evaluation strictly follows the training evaluation process'
                }
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved to: {results_file}")
        
        return accumulated_metrics
    
    def evaluate_with_reference_data(self, save_results: bool = True) -> Dict[str, float]:
        """
        使用参考JSON中的固定数据进行评估
        
        Args:
            save_results: 是否保存详细结果
            
        Returns:
            评估指标字典
        """
        print(f"\n🎯 Starting evaluation with FIXED reference data from JSON")
        print(f"📊 Total reference samples: {len(self.reference_data)}")
        print("=" * 60)
        
        # 生成商品embeddings
        self.generate_item_embeddings()
        
        # 初始化累积指标
        accumulated_metrics = {
            'hit_rate@5': 0.0, 'hit_rate@10': 0.0, 'hit_rate@20': 0.0,
            'ndcg@5': 0.0, 'ndcg@10': 0.0, 'ndcg@20': 0.0
        }
        
        valid_samples = 0
        detailed_results = []
        
        for ref_idx, ref_sample in enumerate(tqdm(self.reference_data, desc="Evaluating with reference data")):
            # 从参考数据中获取信息
            user_id = ref_sample.get('user_id')
            candidate_ids = ref_sample.get('candidate_ids')
            sample_index = ref_sample.get('sample_index')
            
            if not user_id or not candidate_ids:
                print(f"⚠️  Skipping reference sample {ref_idx}: missing user_id or candidate_ids")
                continue
            
            # 选择样本优先级：优先信任 JSON 中的 sample_index，其次根据 user_id 查找
            sample_idx = None
            # 1) 优先使用 JSON 里记录的 sample_index（这与 test_same_with_test 的记录一一对应）
            if isinstance(sample_index, int) and 0 <= sample_index < len(self.test_data):
                sample_idx = sample_index
            else:
                # 2) 回退到 user_id 映射
                if user_id in self.user_id_to_sample:
                    sample_idx = self.user_id_to_sample[user_id]
                else:
                    print(f"⚠️  User {user_id} not found in test dataset; skipping ref #{ref_idx}")
                    continue

            sample = self.test_data[sample_idx]

            # 一致性检查：若 JSON 给了 sample_index 且 user_id 不一致，打印更明确的提示
            if isinstance(sample_index, int) and 0 <= sample_index < len(self.test_data):
                sample_user_id = sample.get('user_id', None)
                if str(sample_user_id) != str(user_id):
                    # 在数据集中查找该 user_id 的所有位置，帮助诊断
                    candidates = self.user_id_to_samples.get(user_id, [])
                    print(
                        f"⚠️  Mismatch at ref #{ref_idx}: JSON(sample_index={sample_index}, user_id={user_id}) "
                        f"but dataset[{sample_idx}].user_id={sample_user_id}. Possible duplicate user entries. "
                        f"All indices for this user: {candidates[:5]}{'...' if len(candidates) > 5 else ''}"
                    )
            
            # 获取历史和目标
            history_ids = sample['history_item_id']
            target_id = sample['item_id']
            
            # 跳过padding token
            if target_id == 0:
                print(f"⚠️  Skipping sample {sample_idx}: target_id is 0")
                continue
            
            # 过滤历史中的padding token
            history_ids = [id for id in history_ids if id != 0]
            history_titles = [self.id2title.get(id, f"Unknown_{id}") for id in history_ids]
            target_title = self.id2title.get(target_id, f"Unknown_{target_id}")
            
            # 使用JSON中固定的候选商品列表
            candidate_items = candidate_ids.copy()
            
            # 验证目标商品是否在候选中
            if target_id not in candidate_items:
                print(f"⚠️  Warning: target_id {target_id} not in candidate_ids for sample {sample_idx}")
                # 可以选择跳过或添加目标商品
                # candidate_items.append(target_id)
                continue
            
            # 验证候选商品的有效性
            valid_candidates = []
            for item_id in candidate_items:
                if item_id in self.item_embeddings:
                    valid_candidates.append(item_id)
                else:
                    print(f"⚠️  Warning: candidate item {item_id} not found in embeddings")
            
            if not valid_candidates:
                print(f"⚠️  Skipping sample {sample_idx}: no valid candidates")
                continue
            
            candidate_items = valid_candidates
            
            # 生成用户推理（与训练时完全一致）
            user_reasoning = self.generate_user_reasoning_training_style(sample)
            
            # 从推理中提取用户embedding（与训练时完全一致）
            user_embedding = self.get_user_embedding_training_style(sample, user_reasoning)
            
            # 计算相似度（与训练时完全一致）
            similarity_scores = self.compute_similarity_training_style(user_embedding, candidate_items)
            
            # 排序获得预测结果
            _, sorted_indices = torch.sort(similarity_scores, descending=True)
            predicted_ranking = [candidate_items[idx] for idx in sorted_indices.cpu().tolist()]
            
            # 计算指标
            sample_metrics = self.calculate_metrics_like_training(predicted_ranking, target_id)
            
            # 累积指标
            for key in accumulated_metrics:
                accumulated_metrics[key] += sample_metrics[key]
            
            valid_samples += 1
            
            # 保存详细结果
            if save_results:
                target_rank = predicted_ranking.index(target_id) + 1 if target_id in predicted_ranking else len(predicted_ranking) + 1
                detailed_results.append({
                    'sample_id': sample_idx,
                    'reference_index': ref_idx,
                    'user_id': user_id,
                    'history_titles': history_titles[-self.user_window_size:],
                    'target_title': target_title,
                    'target_id': target_id,
                    'target_rank': target_rank,
                    'total_candidates': len(candidate_items),
                    'candidate_items': candidate_items,
                    'predicted_ranking': predicted_ranking,
                    'similarity_scores': similarity_scores.cpu().tolist(),
                    'user_reasoning': user_reasoning,
                    'user_reasoning_prompt': getattr(self, 'last_user_reasoning_prompt', None),
                    'metrics': sample_metrics,
                    'reference_data': ref_sample  # 保存原始参考数据用于对比
                })
            
            # 每10个样本输出一次进度
            if valid_samples % 10 == 0:
                current_metrics = {k: v / valid_samples for k, v in accumulated_metrics.items()}
                print(f"\n--- Progress: {valid_samples}/{len(self.reference_data)} samples ---")
                print(f"Current NDCG@10: {current_metrics['ndcg@10']:.6f}")
                print(f"Current Hit Rate@10: {current_metrics['hit_rate@10']:.6f}")
        
        # 计算平均指标
        if valid_samples > 0:
            for key in accumulated_metrics:
                accumulated_metrics[key] /= valid_samples
        
        accumulated_metrics['total_samples'] = valid_samples
        
        # 打印最终结果
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION RESULTS WITH REFERENCE DATA ({valid_samples} samples)")
        print(f"{'='*60}")
        print(f"hit_rate@5:  {accumulated_metrics['hit_rate@5']:.6f}")
        print(f"hit_rate@10: {accumulated_metrics['hit_rate@10']:.6f}")
        print(f"hit_rate@20: {accumulated_metrics['hit_rate@20']:.6f}")
        print(f"ndcg@5:      {accumulated_metrics['ndcg@5']:.6f}")
        print(f"ndcg@10:     {accumulated_metrics['ndcg@10']:.6f}")
        print(f"ndcg@20:     {accumulated_metrics['ndcg@20']:.6f}")
        
        # 保存结果
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f"reference_based_results_{self.model_type}_{valid_samples}samples_{timestamp}.json"
            
            results = {
                'metrics': accumulated_metrics,
                'detailed_results': detailed_results,
                'config': {
                    'checkpoint_path': self.checkpoint_path,
                    'dataset_dir': self.dataset_dir,
                    'model_type': self.model_type,
                    'reference_json_path': self.reference_json_path,
                    'num_samples': valid_samples,
                    'evaluation_method': 'reference_based',
                    'device': self.device,
                    'note': 'This evaluation uses fixed data from reference JSON file'
                }
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved to: {results_file}")
        
        return accumulated_metrics

def main():
    parser = argparse.ArgumentParser(description="Training-consistent checkpoint testing")
    parser.add_argument("--checkpoint_path", type=str, 
                       default="/data2/home/hongdeyao/RRec/checkpoint-test",
                       help="Checkpoint path")
    parser.add_argument("--dataset_dir", type=str, 
                       default="/data/hongdeyao/Musical_Instruments_0_2022-10-2023-10",
                       help="Dataset directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of test samples (ignored if reference_json is provided)")
    parser.add_argument("--model_type", type=str, default="qwen",
                       choices=["qwen", "gemma"],
                       help="Model type")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--reference_json", type=str, default=None,
                       help="Path to reference JSON file with fixed user_id and candidate_ids")
    parser.add_argument("--user_window_size", type=int, default=20,
                       help="How many most recent user history interactions to keep when building prompts")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 检查路径
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint path not found: {args.checkpoint_path}")
        return
    
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        return
    
    # 初始化测试器
    tester = TrainingConsistentTester(
        checkpoint_path=args.checkpoint_path,
        dataset_dir=args.dataset_dir,
        model_type=args.model_type,
        device=args.device,
        reference_json_path=args.reference_json,
        user_window_size=args.user_window_size
    )
    
    # 运行评估
    results = tester.evaluate_training_consistent(
        num_samples=args.num_samples,
        save_results=True
    )
    
    print("\n🎉 Training-consistent evaluation completed successfully!")

if __name__ == "__main__":
    main()
