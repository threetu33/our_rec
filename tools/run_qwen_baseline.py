#!/usr/bin/env python3
"""
处理generated_prompts.json中的prompts，使用本地Qwen2.5-3B-Instruct模型生成回答
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data/hongdeyao/code/RRec/sft/logs/run_qwen_baseline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PromptProcessor:
    def __init__(self, model_path: str, output_file: str):
        """
        初始化处理器
        
        Args:
            model_path: 本地模型路径
            output_file: 输出结果文件路径
        """
        self.model_path = model_path
        self.output_file = output_file
        self.device = "cuda:7"
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"模型路径: {model_path}")
        logger.info(f"输出文件: {output_file}")
        
        # 初始化结果存储
        self.results = {
            "metadata": {
                "model_path": model_path,
                "processing_start_time": datetime.now().isoformat(),
                "total_processed": 0,
                "device": self.device
            },
            "responses": []
        }
        
        # 如果输出文件已存在，加载已有结果
        self._load_existing_results()
        
        # 加载模型和tokenizer
        self._load_model()
    
    def _load_existing_results(self):
        """加载已存在的结果文件，支持断点续传"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    self.results = existing_data
                    logger.info(f"加载已存在的结果文件，已处理 {len(self.results['responses'])} 个prompts")
            except Exception as e:
                logger.warning(f"无法加载已存在的结果文件: {e}")
                # 备份原文件
                if os.path.exists(self.output_file):
                    backup_file = f"{self.output_file}.backup_{int(time.time())}"
                    os.rename(self.output_file, backup_file)
                    logger.info(f"原文件已备份为: {backup_file}")
    
    def _load_model(self):
        """加载模型和tokenizer"""
        try:
            logger.info("开始加载模型和tokenizer...")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map={"": self.device},
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("模型和tokenizer加载完成")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def _generate_response(self, prompt: str) -> str:
        """
        使用模型生成回答
        
        Args:
            prompt: 输入的prompt
            
        Returns:
            生成的回答文本
        """
        try:
            # 构建聊天格式的输入
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 编码输入
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=8192  # 根据模型的最大长度调整
            ).to(self.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            logger.error(f"生成回答时出错: {e}")
            return f"[ERROR] 生成回答失败: {str(e)}"
    
    def _save_results(self):
        """保存结果到文件"""
        try:
            # 更新metadata
            self.results["metadata"]["total_processed"] = len(self.results["responses"])
            self.results["metadata"]["last_update_time"] = datetime.now().isoformat()
            
            # 写入文件
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            raise
    
    def process_prompts(self, input_file: str):
        """
        处理prompts文件
        
        Args:
            input_file: 包含prompts的JSON文件路径
        """
        try:
            # 读取输入文件
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            prompts = data.get('prompts', [])
            total_prompts = len(prompts)
            
            logger.info(f"总共需要处理 {total_prompts} 个prompts")
            
            # 获取已处理的prompt_id集合，避免重复处理
            processed_ids = {item['prompt_id'] for item in self.results['responses']}
            
            processed_count = len(processed_ids)
            logger.info(f"已处理 {processed_count} 个prompts，剩余 {total_prompts - processed_count} 个")
            
            # 逐个处理prompts
            for i, prompt_data in enumerate(prompts):
                prompt_id = prompt_data.get('prompt_id')
                
                # 跳过已处理的prompt
                if prompt_id in processed_ids:
                    continue
                
                full_prompt = prompt_data.get('full_prompt', '')
                
                if not full_prompt:
                    logger.warning(f"Prompt {prompt_id} 的full_prompt为空，跳过")
                    continue
                
                logger.info(f"处理 Prompt {prompt_id} ({i+1}/{total_prompts})")
                
                start_time = time.time()
                
                # 生成回答
                response = self._generate_response(full_prompt)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # 构建结果
                result = {
                    "prompt_id": prompt_id,
                    "sample_id": prompt_data.get('sample_id'),
                    "user_id": prompt_data.get('user_id'),
                    "target_item_id": prompt_data.get('target_item_id'),
                    "candidate_item_id": prompt_data.get('candidate_item_id'),
                    "is_match": prompt_data.get('is_match'),
                    "input_prompt": full_prompt,
                    "model_response": response,
                    "processing_time_seconds": round(processing_time, 2),
                    "timestamp": datetime.now().isoformat()
                }
                
                # 添加到结果列表
                self.results['responses'].append(result)
                
                # 立即保存结果
                self._save_results()
                
                logger.info(f"Prompt {prompt_id} 处理完成，用时 {processing_time:.2f}秒")
                
                # 每10个prompt输出进度
                if (len(self.results['responses']) - processed_count) % 10 == 0:
                    current_processed = len(self.results['responses'])
                    progress = (current_processed / total_prompts) * 100
                    logger.info(f"进度: {current_processed}/{total_prompts} ({progress:.1f}%)")
            
            logger.info("所有prompts处理完成！")
            
        except Exception as e:
            logger.error(f"处理prompts时出错: {e}")
            raise
    
    def get_statistics(self):
        """获取处理统计信息"""
        total_responses = len(self.results['responses'])
        if total_responses == 0:
            return "暂无处理结果"
        
        total_time = sum(item.get('processing_time_seconds', 0) for item in self.results['responses'])
        avg_time = total_time / total_responses if total_responses > 0 else 0
        
        stats = f"""
处理统计信息:
- 总处理数量: {total_responses}
- 总处理时间: {total_time:.2f} 秒
- 平均处理时间: {avg_time:.2f} 秒/prompt
- 输出文件: {self.output_file}
        """
        
        return stats


def main():
    """主函数"""
    # 配置路径
    model_path = "/data/public_checkpoints/huggingface_models/Qwen2.5-3B-Instruct"
    input_file = "/data/hongdeyao/code/RRec/sft/data/test_prompts_check.json"
    output_file = "/data/hongdeyao/code/RRec/sft/data/test_qwen_baseline_prompts_check.json"

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return
    
    # 检查输出目录是否存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")
    
    try:
        # 创建处理器并开始处理
        processor = PromptProcessor(model_path, output_file)
        processor.process_prompts(input_file)
        
        # 输出统计信息
        logger.info(processor.get_statistics())
        
    except KeyboardInterrupt:
        logger.info("用户中断处理")
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
