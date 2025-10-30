#!/usr/bin/env python3
"""
处理generated_prompts.json中的prompts，使用DeepSeek API生成回答
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import random

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data/hongdeyao/code/RRec/sft/logs/run_deepseek_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 线程锁，用于保护文件写入
file_lock = threading.Lock()

class PromptProcessor:
    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        output_file: str,
        max_workers: int = 8,
        prompt_id_min: Optional[int] = None,
        prompt_id_max: Optional[int] = None,
    ):
        """
        初始化处理器
        
        Args:
            api_base_url: API基础URL
            api_key: API密钥
            output_file: 输出结果文件路径
            max_workers: 线程池最大工作线程数
            prompt_id_min: 要处理的最小prompt_id（包含）
            prompt_id_max: 要处理的最大prompt_id（包含）
        """
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.output_file = output_file
        self.max_workers = max_workers
        self.max_retries = 20  # 最大重试次数
        self.prompt_id_min = prompt_id_min
        self.prompt_id_max = prompt_id_max
        self._invalid_prompt_ids_reported: Set[str] = set()
        
        logger.info(f"API URL: {api_base_url}")
        logger.info(f"输出文件: {output_file}")
        logger.info(f"最大并发线程数: {max_workers}")
        if prompt_id_min is not None or prompt_id_max is not None:
            logger.info(
                "Prompt ID范围: %s - %s",
                str(prompt_id_min) if prompt_id_min is not None else "-∞",
                str(prompt_id_max) if prompt_id_max is not None else "+∞",
            )
        else:
            logger.info("Prompt ID范围: 未限制")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(base_url=api_base_url, api_key=api_key)
        self.model_names = ["deepseek-reasoner"]
        
        # 初始化结果存储
        self.results = {
            "metadata": {
                "api_base_url": api_base_url,
                "processing_start_time": datetime.now().isoformat(),
                "total_processed": 0,
                "max_workers": max_workers,
                "prompt_id_min": prompt_id_min,
                "prompt_id_max": prompt_id_max,
                "total_candidates_in_range": None,
            },
            "responses": []
        }
        
        # 如果输出文件已存在，加载已有结果
        self._load_existing_results()
        self.results.setdefault("metadata", {})
        self.results.setdefault("responses", [])

        metadata = self.results["metadata"]
        metadata.setdefault("processing_start_time", datetime.now().isoformat())
        metadata["api_base_url"] = api_base_url
        metadata["max_workers"] = max_workers
        metadata["prompt_id_min"] = prompt_id_min
        metadata["prompt_id_max"] = prompt_id_max
        metadata.setdefault("total_candidates_in_range", None)
        metadata["total_processed"] = len(self.results["responses"])

    def _parse_prompt_id(self, prompt_id: Any) -> Optional[int]:
        """将prompt_id解析为整数，如果失败则返回None"""
        if prompt_id is None:
            return None
        try:
            return int(str(prompt_id).strip())
        except (ValueError, TypeError):
            return None

    def _is_prompt_id_within_bounds(self, prompt_id_value: Any) -> bool:
        """判断给定的prompt_id值是否满足当前范围限制，不产生日志"""
        if self.prompt_id_min is None and self.prompt_id_max is None:
            return True

        prompt_id_int = self._parse_prompt_id(prompt_id_value)
        if prompt_id_int is None:
            return False

        if self.prompt_id_min is not None and prompt_id_int < self.prompt_id_min:
            return False

        if self.prompt_id_max is not None and prompt_id_int > self.prompt_id_max:
            return False

        return True

    def _is_prompt_in_range(self, prompt_data: Dict[str, Any]) -> bool:
        """判断prompt是否在指定的ID范围内"""
        if self.prompt_id_min is None and self.prompt_id_max is None:
            return True

        prompt_id = prompt_data.get("prompt_id")
        if not self._is_prompt_id_within_bounds(prompt_id):
            prompt_id_int = self._parse_prompt_id(prompt_id)
            if prompt_id_int is not None:
                return False

            key = str(prompt_id)
            if key not in self._invalid_prompt_ids_reported:
                logger.warning(
                    "Prompt %s 的prompt_id无法转换为整数，已跳过。", key
                )
                self._invalid_prompt_ids_reported.add(key)
            return False

        return True
    
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
    
    def _generate_response_with_retry(self, prompt: str, prompt_id: str) -> Dict[str, Any]:
        """
        使用API生成回答，带重试机制
        
        Args:
            prompt: 输入的prompt
            prompt_id: prompt的ID，用于日志
            
        Returns:
            包含回答和思考过程的字典: {"answer": str, "reasoning_content": str|None}
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                # 随机选择一个模型（如果需要的话）
                model_name = self.model_names[0]
                
                # 调用API
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
                
                # 提取回答内容
                answer = response.choices[0].message.content
                reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
                
                if attempt > 1:
                    logger.info(f"Prompt {prompt_id} 在第 {attempt} 次尝试后成功")
                
                return {
                    "answer": answer.strip() if answer else "[ERROR] 空回答",
                    "reasoning_content": reasoning_content
                }
                
            except Exception as e:
                logger.warning(f"Prompt {prompt_id} 第 {attempt} 次尝试失败: {str(e)}")
                
                if attempt < self.max_retries:
                    # 指数退避重试
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"等待 {wait_time:.2f} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Prompt {prompt_id} 在 {self.max_retries} 次尝试后仍然失败")
                    return {
                        "answer": f"[ERROR] API调用失败，已重试{self.max_retries}次: {str(e)}",
                        "reasoning_content": None
                    }
    
    def _process_single_prompt(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个prompt
        
        Args:
            prompt_data: 包含prompt信息的字典
            
        Returns:
            处理结果字典
        """
        prompt_id = prompt_data.get('prompt_id')
        full_prompt = prompt_data.get('full_prompt', '')
        
        if not full_prompt:
            logger.warning(f"Prompt {prompt_id} 的full_prompt为空，跳过")
            return None
        
        logger.info(f"开始处理 Prompt {prompt_id}")
        
        start_time = time.time()
        
        # 生成回答
        response_data = self._generate_response_with_retry(full_prompt, prompt_id)
        
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
            "model_response": response_data["answer"],
            "model_thinking": response_data["reasoning_content"],
            "processing_time_seconds": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Prompt {prompt_id} 处理完成，用时 {processing_time:.2f}秒")
        
        return result
    
    def _save_results(self):
        """保存结果到文件（线程安全）"""
        try:
            with file_lock:
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
        并行处理prompts文件
        
        Args:
            input_file: 包含prompts的JSON文件路径
        """
        try:
            # 读取输入文件
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            prompts = data.get('prompts', [])
            total_prompts_all = len(prompts)
            
            if total_prompts_all == 0:
                logger.info("输入文件中没有prompts")
                return

            prompts_in_range = [p for p in prompts if self._is_prompt_in_range(p)]
            total_prompts = len(prompts_in_range)

            if total_prompts == 0:
                logger.info(
                    "没有符合指定Prompt ID范围的prompts。范围: %s - %s",
                    str(self.prompt_id_min) if self.prompt_id_min is not None else "-∞",
                    str(self.prompt_id_max) if self.prompt_id_max is not None else "+∞",
                )
                return

            skipped_by_range = total_prompts_all - total_prompts

            logger.info(f"输入总计 {total_prompts_all} 个prompts，其中 {total_prompts} 个满足范围条件")
            if skipped_by_range > 0:
                logger.info(f"有 {skipped_by_range} 个prompts因不满足Prompt ID范围而被跳过")

            # 记录范围内的总数
            self.results["metadata"]["total_candidates_in_range"] = total_prompts
            
            # 获取已处理的prompt_id集合，避免重复处理
            processed_ids = {item['prompt_id'] for item in self.results['responses']}
            
            # 过滤出未处理的prompts
            unprocessed_prompts = [
                p for p in prompts_in_range if p.get('prompt_id') not in processed_ids
            ]
            
            processed_count = len(processed_ids)
            processed_in_range = sum(
                1
                for item in self.results['responses']
                if self._is_prompt_id_within_bounds(item.get('prompt_id'))
            )
            remaining_count = len(unprocessed_prompts)
            
            logger.info(f"当前范围内已处理 {processed_in_range} 个prompts，剩余 {remaining_count} 个")
            if processed_count != processed_in_range:
                logger.info(
                    "结果文件中包含 %d 个不在当前范围内的历史记录",
                    processed_count - processed_in_range,
                )
            
            if remaining_count == 0:
                logger.info("所有prompts已处理完成！")
                return
            
            # 使用线程池并行处理
            completed_count = 0
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_prompt = {
                    executor.submit(self._process_single_prompt, prompt_data): prompt_data
                    for prompt_data in unprocessed_prompts
                }
                
                # 处理完成的任务
                for future in as_completed(future_to_prompt):
                    try:
                        result = future.result()
                        
                        if result is not None:
                            # 线程安全地添加结果并保存
                            with file_lock:
                                self.results['responses'].append(result)
                            
                            # 立即保存结果
                            self._save_results()
                        
                        completed_count += 1
                        
                        # 每10个prompt输出进度
                        if completed_count % 10 == 0:
                            current_total_in_range = sum(
                                1
                                for item in self.results['responses']
                                if self._is_prompt_id_within_bounds(item.get('prompt_id'))
                            )
                            progress = (current_total_in_range / total_prompts) * 100
                            logger.info(
                                f"进度: {current_total_in_range}/{total_prompts} ({progress:.1f}%)"
                            )
                        
                    except Exception as e:
                        prompt_data = future_to_prompt[future]
                        prompt_id = prompt_data.get('prompt_id', 'unknown')
                        logger.error(f"处理Prompt {prompt_id}时发生异常: {e}")
            
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="批量调用DeepSeek接口生成回答，支持按prompt_id范围筛选"
    )
    parser.add_argument(
        "--prompt-id-min",
        type=int,
        default=0,
        help="只处理prompt_id大于等于该值的记录",
    )
    parser.add_argument(
        "--prompt-id-max",
        type=int,
        default=400000,
        help="只处理prompt_id小于等于该值的记录",
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 配置路径和API参数
    api_base_url = "http://115.182.62.174:18888/v1"
    api_key = "sk-ZUK05LeE0U3FVQAp6e5eCd3628774c4181D513786bD3B901"
    input_file = "/data/hongdeyao/code/RRec/sft/data/sft_train_movies_prompts_10000.json"
    output_file = "/data/hongdeyao/code/RRec/sft/data/sft_deepseek_movies_response_10000.json"
    max_workers = 8  # 并发线程数，可根据API限制调整

    prompt_id_min = args.prompt_id_min
    prompt_id_max = args.prompt_id_max

    if (
        prompt_id_min is not None
        and prompt_id_max is not None
        and prompt_id_min > prompt_id_max
    ):
        logger.error("prompt_id_min 不能大于 prompt_id_max")
        return

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    # 检查输出目录是否存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")
    
    try:
        # 创建处理器并开始处理
        processor = PromptProcessor(
            api_base_url,
            api_key,
            output_file,
            max_workers,
            prompt_id_min=prompt_id_min,
            prompt_id_max=prompt_id_max,
        )
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
