# Global-sequence PPO-clip (with Value) for PL-ranking over LLM embeddings.
# Pipeline: Prompt -> LLM -> ScoreHead -> PL logits -> sample full ranking (length K_eval)
# Reward = NDCG@K_eval (global). PPO-clip on the whole sequence logprob.
# Backprop to LLM + ScoreHead (actor) and ValueHead (critic).
from __future__ import annotations
import math
import random
import argparse
import json, glob, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributions import Categorical
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datasets as hfds
from transformers import AutoTokenizer
from datetime import datetime

# ===== 你的 LLM 封装（保持）=====
from models.qwen_models import Qwen2RRecCasualLM


# =========================
# Prompt 模板 & 文本处理 (与 sft_data.py 保持一致)
# =========================
# 根据数据集类型选择不同的 Prompt 模板
BINARY_PROMPT_TMPL_MOVIES = (
    "You are a recommendation assistant. Analyze the user's history in depth and decide whether to recommend the candidate item. "
    "First analyze then Answer inside {emb_token} and {emb_end_token}."
    "### User history (most recent last)\n"
    "{user_text}\n\n"
    "### Candidate item\n"
    "{item_text}\n"
)

BINARY_PROMPT_TMPL_OTHER = (
    "You are a recommendation assistant. Analyze the user's history in depth and decide whether to recommend the candidate item. "
    "First analyze then Answer inside {emb_token} and {emb_end_token}."
    "### User history (most recent last)\n"
    "{user_text}\n\n"
    "### Candidate item\n"
    "{item_text}\n"
)

# 初始化为 None，将在 parse_args 后根据 dataset_type 参数设置
BINARY_PROMPT_TMPL = None
# BINARY_PROMPT_TMPL = (
#     "Please judge how well the candidate item matches the user's preferences based on the history below and the candidate item information.Generate a response with three parts: #Initial reasoning: #Self-check: #Final conclusion and suggestion: .End with a clear \"yes\" or \"no\" judgment indicating whether the candidate item matches the user's preferences, enclosed within `{emb_token}` and `{emb_end_token}`."
#     "# HISTORY:\n\n"
#     "historical musical instruments purchases and ratings (out of 5):\n"
#     "{user_text}\n\n"
#     "# CANDIDATE ITEM:\n\n"
#     "{item_text}\n\n"
# )
# BINARY_PROMPT_TMPL = (
#     "Based on the user's history, please judge how well the candidate item matches the user's preferences based on the history below and the candidate item information. "
#     "Produce a **single coherent paragraph** that contains the following three parts **in order** (with parts labeled by a single `#` as explained): initial reasoning, self-check/evaluation, final conclusion. "
#     # "**Mark each part by placing a single `#` immediately before its label** (for example `#Initial reasoning:`, `#Self-check:`, `#Final conclusion and suggestion:`). "
#     # "Do not add other `#` symbols. Please keep the entire paragraph to **roughly 200–400 words**. "
#     # "It is acceptable to draw reasonable inferences from the provided information. "
#     "Finally, conclude with a clear \"yes\" or \"no\" judgment indicating whether the candidate item matches the user's preferences, enclosed within `{emb_token}` and `{emb_end_token}`.\n\n"
#     # "* `#Initial reasoning:` — perform a normal, thorough reasoning step that **fully considers relevant factors**.\n"
#     # "* `#Self-check:` — evaluate the initial reasoning: explicitly identify any mistakes, overlooked evidence, uncertainties, or over-weighted signals in the first pass; either confirm the initial reasoning or explain why and how you would modify it. This is a short critical reflection on the first pass.\n"
#     # "* `#Final conclusion and suggestion:` — combine the outcomes of the first two parts and present a succinct final reasoning and conclusion.\n\n"
#     "# HISTORY:\n\n"
#     "historical musical instruments purchases and ratings (out of 5):\n"
#     "{user_text}\n\n"
#     "# CANDIDATE ITEM:\n\n"
#     "{item_text}\n\n"
#     # "# TASK:\n\n"
#     # "Based on the user's history above, produce the single-paragraph output described (with parts labeled by a single `#` as explained), followed by a final yes/no judgment within `{emb_token}` tags.\n"
# )
# BINARY_PROMPT_TMPL = (
#     "Please judge how well the candidate item matches the user's preferences based on the history below and the candidate item information. "
#     "Produce a **single coherent paragraph** that contains the following three parts **in order**: initial reasoning, self-check/evaluation, and a final reasoning. "
#     "**Mark each part by placing a single `#` immediately before its label** (for example `#Initial reasoning:`, `#Self-check:`, `#Final conclusion and suggestion:`). "
#     "Do not add other `#` symbols. Please keep the entire paragraph to **roughly 200–400 words**. "
#     "It is acceptable to draw reasonable inferences from the provided information. "
#     "Finally, conclude with a clear \"yes\" or \"no\" judgment indicating whether the candidate item matches the user's preferences, enclosed within `{emb_token}` and `{emb_end_token}`.\n\n"
#     "* `#Initial reasoning:` — perform a normal, thorough reasoning step that **fully considers relevant factors**.\n"
#     "* `#Self-check:` — evaluate the initial reasoning: explicitly identify any mistakes, overlooked evidence, uncertainties, or over-weighted signals in the first pass; either confirm the initial reasoning or explain why and how you would modify it. This is a short critical reflection on the first pass.\n"
#     "* `#Final conclusion and suggestion:` — combine the outcomes of the first two parts and present a succinct final reasoning and conclusion.\n\n"
#     "# HISTORY:\n\n"
#     "historical musical instruments purchases and ratings (out of 5):\n"
#     "{user_text}\n\n"
#     "# CANDIDATE ITEM:\n\n"
#     "{item_text}\n\n"
#     "# TASK:\n\n"
#     "Based on the user's history above, produce the single-paragraph output described (with parts labeled by a single `#` as explained), followed by a final yes/no judgment within `{emb_token}` tags.\n"
# )
DESCRIPTION_MAX_LEN = 100
SEPARATOR = "\n\n-----\n\n"

# 新增导入
from tqdm import tqdm
from transformers import GenerationConfig

# 提取 <answer> ... </answer> 之间内容的小工具
def extract_between(text: str, start_tag: str, end_tag: str) -> str:
    s = text.find(start_tag)
    if s == -1: return text
    s += len(start_tag)
    e = text.find(end_tag, s)
    if e == -1: return text[s:]
    return text[s:e].strip()

@torch.no_grad()
def preview_answers(model: Qwen2RRecCasualLM, tok, prompter: BinaryItemPrompter,
                    raw_prompts_20: List[List[str]],
                    preview_n: int = 1, max_new_tokens: int = 128, max_len_per_item: int = 512,
                    log_path: Optional[str] = None, step: Optional[int] = None):
    """
    从本 batch 的 raw_prompts_20 中抽取若干条，调用 model.generate 打印真实输出。
    使用与训练时相同的截断逻辑。
    """
    if preview_n <= 0 or len(raw_prompts_20) == 0:
        return
    device = next(model.parameters()).device
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True, temperature=0.7, top_p=0.9,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )
    # 逐样本最多打印一条；最多打印 preview_n 个样本
    stats: List[Dict[str, Any]] = []
    for sample_prompts in raw_prompts_20[:preview_n]:
        if not sample_prompts: 
            continue
        idx = random.randrange(len(sample_prompts))
        text = sample_prompts[idx]
        
        # ★ 应用与训练时相同的截断逻辑 ★
        inputs = tok(text, return_tensors="pt", add_special_tokens=False, 
                    truncation=True, max_length=max_len_per_item).to(device)
        
        # 检查是否发生了截断
        input_len = inputs['input_ids'].size(1)
        was_truncated = (input_len >= max_len_per_item)
        
        out_ids = model.generate(**inputs, generation_config=gen_cfg)
        out_txt = tok.decode(out_ids[0], skip_special_tokens=False)
        ans = extract_between(out_txt, prompter.emb_token, prompter.emb_end_token)
        total_len = out_ids.size(1)
        new_len = max(0, total_len - input_len)
        ans_token_len = len(tok(ans, add_special_tokens=False)["input_ids"]) if ans else 0
        rec = {
            "step": int(step) if step is not None else None,
            "input_tokens": int(input_len),
            "total_tokens": int(total_len),
            "new_tokens": int(new_len),
            "answer_tokens": int(ans_token_len),
            "was_truncated": bool(was_truncated),
        }
        stats.append(rec)
        
        print("---- LLM Preview ----")
        print(f"ORIGINAL PROMPT length: {len(tok.encode(text, add_special_tokens=False))} tokens")
        print(f"TRUNCATED TO: {input_len} tokens (max_len_per_item={max_len_per_item})")
        print(f"WAS TRUNCATED: {'YES ⚠️' if was_truncated else 'NO'}")
        print(f"TOTAL TOKENS (incl. prompt): {total_len}")
        print(f"NEW TOKENS (generated): {new_len}")
        print(f"TOKENS inside tags {prompter.emb_token}...{prompter.emb_end_token}: {ans_token_len}")
        print(f"\nTRUNCATED PROMPT (decoded back):\n{tok.decode(inputs['input_ids'][0])}\n")
        print("="*80)
        print(f"RAW OUTPUT (last 2000 chars):\n{out_txt[-2000:]}")
        print("="*80)
        print(f"ANSWER between tags `{prompter.emb_token}` and `{prompter.emb_end_token}`: {ans}")
        print("---------------------")

    # 打印统计摘要并可选写入日志文件
    if stats:
        new_lens = [s["new_tokens"] for s in stats]
        avg_new = sum(new_lens) / len(new_lens)
        print(f"[preview token stats] new_tokens avg={avg_new:.1f}, min={min(new_lens)}, max={max(new_lens)} over {len(new_lens)} samples")
        if log_path:
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a") as f:
                    for s in stats:
                        json.dump(s, f, ensure_ascii=False)
                        f.write("\n")
            except Exception as e:
                print(f"[warn] failed to write token stats to {log_path}: {e}")

def _format_timedelta(delta):
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    minutes += seconds / 60
    if days > 0:
        hours += minutes / 60
        s = f"{days}d {hours:.1f}h"
    else:
        s = f"{hours}h {minutes:.1f}min" if hours > 0 else f"{minutes:.1f}min"
    return s + " ago"


def _timestamps_to_human(history_ts_ms: List[int], ts_ms: int) -> List[str]:
    history_dt = [datetime.fromtimestamp(t/1000) for t in history_ts_ms]
    t = datetime.fromtimestamp(ts_ms/1000)
    deltas = [t - h for h in history_dt]
    return [_format_timedelta(d) for d in deltas]


def item_record_to_dict(item_rec: Dict[str, Any]) -> Dict[str, Any]:
    title = item_rec.get("title") or item_rec.get("item_title") or "Unknown Item"
    avg = item_rec.get("average_rating", 0.0)
    buyers = item_rec.get("rating_number", 0)
    desc = item_rec.get("description", "")
    if isinstance(desc, list):
        desc = "" if len(desc) == 0 else " ".join(desc[::-1])
    words = desc.split()
    if len(words) > DESCRIPTION_MAX_LEN:
        desc = " ".join(words[:DESCRIPTION_MAX_LEN]) + "..."
    
    result = {
        "title": title,
        "average_rating": avg,
        "rating_number": buyers,
        "description": desc,
    }
    
    # 添加新的字段（如果存在）
    if "main_category" in item_rec:
        result["main_category"] = item_rec["main_category"]
    if "price" in item_rec:
        result["price"] = item_rec["price"]
    if "store" in item_rec:
        result["store"] = item_rec["store"]
    if "categories" in item_rec:
        result["categories"] = item_rec["categories"]
    if "features" in item_rec:
        result["features"] = item_rec["features"]
        
    return result


def build_user_text_movies(sequence: Dict[str, Any],
                           id2title: Dict[int, str],
                           id2info: Dict[int, Dict[str, Any]] = None,
                           win_size: int = 10) -> str:
    """构建电影数据集的用户历史文本（包含类型、年份、IMDb评分等信息）"""
    titles = sequence.get("history_item_title", None)
    if not titles:
        ids = sequence.get("history_item_id", []) or []
        titles = [id2title.get(int(i), f"Item#{int(i)}") for i in ids]

    ratings = sequence.get("history_rating", None)
    history_ts = sequence.get("history_timestamp", None)
    ts = sequence.get("timestamp", None)
    history_ids = sequence.get("history_item_id", [])
    dset_len = len(titles)

    if history_ts is not None and ts is not None:
        human_times = _timestamps_to_human(history_ts, ts)
        if len(human_times) != dset_len:
            human_times = ["t0"] * dset_len
    else:
        human_times = ["t0"] * dset_len

    start = 0 if dset_len <= win_size else dset_len - win_size
    lines = []
    for i in range(start, dset_len):
        title_i = titles[i]
        rating_i = ratings[i] if (ratings and i < len(ratings)) else ""
        
        # 构建信息字符串，包含评分
        info_parts = []
        if rating_i:
            info_parts.append(f"Rating: {rating_i}")
            
        # 如果有物品详细信息映射，添加额外信息
        if id2info and i < len(history_ids):
            item_id = int(history_ids[i])
            item_info = id2info.get(item_id, {})
            
            # 添加电影类型信息（从categories字段提取真实的电影类型）
            categories = item_info.get("categories", [])
            if categories:
                # 筛选出真正的电影类型，排除无用的类别
                genre_categories = []
                skip_categories = {}
                
                for cat in categories[:3]:  # 只取前3个类别
                    if cat and cat.strip() and cat not in skip_categories:
                        genre_categories.append(cat)
                
                if genre_categories:
                    info_parts.append(f"Genre: {', '.join(genre_categories)}")
            
            # 添加年份信息（从features字段提取）
            features = item_info.get("features", [])
            year = None
            for feat in features[:5]:  # 检查前5个特征
                if feat and feat.isdigit() and 1900 <= int(feat) <= 2030:
                    year = feat
                    break
            if year:
                info_parts.append(f"Year: {year}")
            
            # 添加IMDb评分（从features字段提取）
            imdb_rating = None
            for feat in features[:5]:
                if feat and feat.startswith("IMDb "):
                    imdb_rating = feat.replace("IMDb ", "")
                    break
            if imdb_rating:
                info_parts.append(f"IMDb: {imdb_rating}")
                
        
        info_str = " | ".join(info_parts) if info_parts else ""
        if info_str:
            lines.append(f"{human_times[i]}: [{title_i}] ({info_str})")
        else:
            lines.append(f"{human_times[i]}: [{title_i}]")
    return "\n".join(lines)


def build_user_text_other(sequence: Dict[str, Any],
                          id2title: Dict[int, str],
                          win_size: int = 10) -> str:
    """构建其他数据集的用户历史文本（简单格式）"""
    titles = sequence.get("history_item_title", None)
    if not titles:
        ids = sequence.get("history_item_id", []) or []
        titles = [id2title.get(int(i), f"Item#{int(i)}") for i in ids]

    ratings = sequence.get("history_rating", None)
    history_ts = sequence.get("history_timestamp", None)
    ts = sequence.get("timestamp", None)
    dset_len = len(titles)

    if history_ts is not None and ts is not None:
        human_times = _timestamps_to_human(history_ts, ts)
        if len(human_times) != dset_len:
            human_times = ["t0"] * dset_len
    else:
        human_times = ["t0"] * dset_len

    start = 0 if dset_len <= win_size else dset_len - win_size
    lines = []
    for i in range(start, dset_len):
        title_i = titles[i]
        rating_i = ratings[i] if (ratings and i < len(ratings)) else ""
        lines.append(f"{human_times[i]}: [{title_i}] ({rating_i})")
    return "\n".join(lines)


def build_user_text(sequence: Dict[str, Any],
                    id2title: Dict[int, str],
                    id2info: Dict[int, Dict[str, Any]] = None,
                    win_size: int = 10,
                    dataset_type: str = "movies") -> str:
    """根据数据集类型构建用户历史文本"""
    if dataset_type == "movies":
        return build_user_text_movies(sequence, id2title, id2info, win_size)
    else:
        return build_user_text_other(sequence, id2title, win_size)


def build_item_text_movies(item: Dict[str, Any]) -> str:
    """构建电影数据集的候选物品详细信息文本，包含更多字段"""
    lines = [f"Title: {item['title']}"]
    
    # 基本评分信息
    lines.append(f"Average Rating: {item['average_rating']}")
    lines.append(f"Number of Ratings: {item['rating_number']}")
    
    # 电影类型信息（从categories字段提取真实的电影类型）
    if 'categories' in item and item['categories']:
        genre_categories = []
        skip_categories = {}
        
        for cat in item['categories'][:3]:  # 只取前3个类别
            if cat and cat.strip() and cat not in skip_categories:
                genre_categories.append(cat)
        
        if genre_categories:
            lines.append(f"Genres: {', '.join(genre_categories)}")
    
    # 年份和IMDb评分信息（从features字段提取）
    if 'features' in item and item['features']:
        year = None
        imdb_rating = None
        duration = None
        
        for feat in item['features'][:5]:  # 检查前5个特征
            if feat:
                # 提取年份
                if feat.isdigit() and 1900 <= int(feat) <= 2030:
                    year = feat
                # 提取IMDb评分
                elif feat.startswith("IMDb "):
                    imdb_rating = feat.replace("IMDb ", "")
                # 提取时长
                elif "h" in feat and "min" in feat:
                    duration = feat
        
        if year:
            lines.append(f"Year: {year}")
        if imdb_rating:
            lines.append(f"IMDb Rating: {imdb_rating}")
        if duration:
            lines.append(f"Duration: {duration}")

    # 描述信息
    lines.append(f"Description: {item['description']}")
    
    return "\n".join(lines)


def build_item_text_other(item: Dict[str, Any]) -> str:
    """构建其他数据集的候选物品信息文本（简单格式）"""
    return (
        f"Title: {item['title']}\n"
        f"Average Rating: {item['average_rating']}\n"
        f"Number of ratings: {item['rating_number']}\n"
        f"Description: {item['description']}"
    )


def build_item_text(item: Dict[str, Any], dataset_type: str = "movies") -> str:
    """根据数据集类型构建候选物品的详细信息文本"""
    if dataset_type == "movies":
        return build_item_text_movies(item)
    else:
        return build_item_text_other(item)


class BinaryItemPrompter:
    def __init__(self, tokenizer, emb_token: str = "", emb_end_token: str = "", dataset_type: str = "movies"):
        self.tok = tokenizer
        self.emb_token = emb_token or getattr(self.tok, "generation_end", "<answer>")
        self.emb_end_token = emb_end_token or "</answer>"
        self.dataset_type = dataset_type

    def _apply_template(self, text: str, add_generation_prompt: bool = True) -> str:
        if hasattr(self.tok, "apply_chat_template"):
            return self.tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        return text

    def build_single_prompt(self, user_text: str, item_text: str) -> str:
        raw = BINARY_PROMPT_TMPL.format(
            emb_token=self.emb_token,
            emb_end_token=self.emb_end_token,
            user_text=user_text,
            item_text=item_text,
        )
        return self._apply_template(raw, add_generation_prompt=True)

    def build_20_prompts(self, user_text: str, candidates: List[Dict[str, Any]]) -> List[str]:
        return [self.build_single_prompt(user_text, build_item_text(item, self.dataset_type)) for item in candidates]


# =========================
# 数据集读取 & 负采样（保持）
# =========================
class NegativeSampler:
    def __init__(self, item_info: hfds.Dataset, item_id_key: str = "item_id"):
        self.id2row: Dict[int, Dict[str, Any]] = {}
        for row in item_info:
            iid = int(row[item_id_key])
            self.id2row[iid] = row
        self.valid_ids = [iid for iid in self.id2row.keys() if iid != 0]

    def item_row(self, iid: int) -> Dict[str, Any]:
        return self.id2row[int(iid)]

    def sample_candidates(self, pos_item_id: int, K: int) -> Tuple[List[int], int]:
        pool = [i for i in self.valid_ids if i != int(pos_item_id)]
        assert len(pool) >= K - 1, "负样本数量不足，请减小 K 或扩大物品池"
        negs = random.sample(pool, K - 1)
        pos_idx = random.randint(0, K - 1)
        ids = negs
        ids.insert(pos_idx, int(pos_item_id))
        return ids, pos_idx


class RecFromHFUserDataset(Dataset):
    def __init__(self,
                 split_ds: hfds.Dataset,
                 item_info: hfds.Dataset,
                 K: int = 20,
                 user_win_size: int = 10,
                 seed: int = 42,
                 dataset_type: str = "movies"):
        self.split_ds = split_ds
        self.item_info = item_info
        self.K = K
        self.user_win_size = user_win_size
        self.base_seed = seed
        self.dataset_type = dataset_type

        self.sampler = NegativeSampler(item_info, item_id_key="item_id")
        self.id2title = {}
        self.id2info = {}  # 新增：存储完整的物品信息（仅电影数据集使用）
        
        for row in item_info:
            iid = int(row["item_id"])
            title = (row.get("title") or row.get("item_title") or f"Item#{iid}").strip()
            self.id2title[iid] = title
            
            # 只有电影数据集才存储详细的物品信息
            if self.dataset_type == "movies":
                self.id2info[iid] = {
                    "main_category": row.get("main_category", ""),
                    "categories": row.get("categories", []),
                    "features": row.get("features", []),  # 新增：包含年份、IMDb评分等信息
                    "price": row.get("price", None),
                    "store": row.get("store", ""),
                    "average_rating": row.get("average_rating", 0.0),
                    "rating_number": row.get("rating_number", 0),
                }

    def __len__(self): return len(self.split_ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        random.seed(self.base_seed + idx)
        ex = self.split_ds[idx]
        pos_iid = int(ex["item_id"])
        
        if self.dataset_type == "movies":
            user_text = build_user_text(ex, self.id2title, self.id2info, 
                                      win_size=self.user_win_size, dataset_type=self.dataset_type)
        else:
            user_text = build_user_text(ex, self.id2title, 
                                      win_size=self.user_win_size, dataset_type=self.dataset_type)
        
        cand_ids, pos_idx = self.sampler.sample_candidates(pos_iid, self.K)

        candidates: List[Dict[str, Any]] = []
        for iid in cand_ids:
            row = self.sampler.item_row(iid)
            candidates.append(item_record_to_dict(row))

        labels = [0] * self.K
        labels[pos_idx] = 1

        return {
            "user_text": user_text, 
            "candidates": candidates, 
            "labels": labels, 
            "candidate_ids": cand_ids,
            "sample_index": idx,
            "raw_sample": ex  # 保存原始样本数据
        }


# =========================
# Collate（保持）
# =========================
@dataclass
class PackedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    item_offsets: List[List[Tuple[int, int]]]
    pad_left: List[int]
    labels: torch.Tensor
    raw_prompts_20: List[List[str]]
    candidate_ids: List[List[int]]  # 新增：候选item ID列表
    sample_indices: List[int]  # 新增：样本索引
    raw_samples: List[Dict[str, Any]]  # 新增：原始样本数据


class PackedCollate:
    def __init__(self, tokenizer, prompter: BinaryItemPrompter, max_len_per_item: int = 512, K: int = 20):
        self.tok = tokenizer
        self.prompter = prompter
        self.max_len_per_item = max_len_per_item
        self.sep_ids = self.tok(SEPARATOR, add_special_tokens=False)["input_ids"]
        self.K = K

    def __call__(self, batch: List[Dict[str, Any]]) -> PackedBatch:
        input_ids_batch, attn_mask_batch = [], []
        offsets_batch, labels_batch, raw_prompts_20, candidate_ids_batch = [], [], [], []
        sample_indices_batch, raw_samples_batch = [], []

        for ex in batch:
            user_text = ex["user_text"]
            candidates = ex["candidates"]
            labels = ex["labels"]
            candidate_ids = ex.get("candidate_ids", [])
            sample_index = ex.get("sample_index", -1)
            raw_sample = ex.get("raw_sample", {})
            assert len(candidates) == self.K and len(labels) == self.K

            prompts = self.prompter.build_20_prompts(user_text, candidates)
            raw_prompts_20.append(prompts)

            toks = self.tok(prompts, add_special_tokens=False, truncation=True,
                            padding=False, max_length=self.max_len_per_item)
            concat_ids, concat_mask, offsets = [], [], []
            cur = 0
            for ids, mask in zip(toks["input_ids"], toks["attention_mask"]):
                start = cur
                concat_ids.extend(ids); concat_mask.extend(mask)
                cur += len(ids)
                end = cur
                offsets.append((start, end))
                concat_ids.extend(self.sep_ids)
                concat_mask.extend([1] * len(self.sep_ids))
                cur += len(self.sep_ids)
            if len(self.sep_ids) > 0:
                concat_ids = concat_ids[:-len(self.sep_ids)]
                concat_mask = concat_mask[:-len(self.sep_ids)]

            input_ids_batch.append(concat_ids)
            attn_mask_batch.append(concat_mask)
            offsets_batch.append(offsets)
            labels_batch.append(labels)
            candidate_ids_batch.append(candidate_ids)
            sample_indices_batch.append(sample_index)
            raw_samples_batch.append(raw_sample)

        pad_id = self.tok.pad_token_id
        max_len = max(len(x) for x in input_ids_batch)
        input_ids_pad, attn_mask_pad, pad_left = [], [], []
        for ids, mask in zip(input_ids_batch, attn_mask_batch):
            pad_len = max_len - len(ids)
            pad_left.append(pad_len)
            input_ids_pad.append([pad_id] * pad_len + ids)
            attn_mask_pad.append([0] * pad_len + mask)

        input_ids = torch.tensor(input_ids_pad, dtype=torch.long)
        attention_mask = torch.tensor(attn_mask_pad, dtype=torch.long)
        labels = torch.tensor(labels_batch, dtype=torch.long)

        return PackedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            item_offsets=offsets_batch,
            pad_left=pad_left,
            labels=labels,
            raw_prompts_20=raw_prompts_20,
            candidate_ids=candidate_ids_batch,
            sample_indices=sample_indices_batch,
            raw_samples=raw_samples_batch,
        )


# =========================
# 模型：ScoreHead + ValueHead + LLM 封装
# =========================
class ScoreHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class ValueHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


class PolicyValue(nn.Module):
    """
    - encode_items_batched: 从 packed 中切出每个候选段，右pad，过 LLM，拿最后 token 表征
    - score_items: 把 [B*K, D] -> logits
    - build_state_embed: query 级别的状态嵌入（用候选平均 + 剩余比例等）
    """
    def __init__(self, llm: Qwen2RRecCasualLM, score_head: ScoreHead, value_head: ValueHead,
                 pair_micro_bs: int = 64, pad_token_id: Optional[int] = None, freeze_llm: bool = False):
        super().__init__()
        self.llm = llm
        self.score_head = score_head
        self.value_head = value_head
        self.pair_micro_bs = pair_micro_bs
        self.pad_token_id = pad_token_id
        if freeze_llm:
            for p in self.llm.parameters(): p.requires_grad = False

    def _encode_segments(self, seg_list: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        pad_id = self.pad_token_id
        assert pad_id is not None, "pad_token_id must be set"
        use_amp = (self.llm.dtype in (torch.float16, torch.bfloat16))
        amp_dtype = self.llm.dtype if isinstance(self.llm.dtype, torch.dtype) else None
        for st in range(0, len(seg_list), self.pair_micro_bs):
            chunk = seg_list[st: st+self.pair_micro_bs]
            max_len = max(t.size(0) for t in chunk)
            padded = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=device)
            attn   = torch.zeros_like(padded)
            for i, t in enumerate(chunk):
                L = t.size(0)
                padded[i, :L] = t.to(device)
                attn[i, :L] = 1
            with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                last_h = self.llm(
                    input_ids=padded,
                    attention_mask=attn,
                    return_causal_output=False,
                    return_with_last_hidden_states=True
                )
            outs.append(last_h)
        return torch.cat(outs, dim=0)  # [N,D]

    def encode_items_batched(self, input_ids, attention_mask, item_offsets, pad_left):
        device = input_ids.device
        B = input_ids.size(0)
        segs: List[torch.Tensor] = []
        segs_per_sample: List[int] = []
        for b in range(B):
            base = pad_left[b]
            count = 0
            for (st, ed) in item_offsets[b]:
                s = base + st
                e = base + ed
                segs.append(input_ids[b, s:e])
                count += 1
            segs_per_sample.append(count)
        H_all = self._encode_segments(segs, device)  # [B*K, D]
        H_per_sample: List[torch.Tensor] = []
        cur = 0
        for n in segs_per_sample:
            H_per_sample.append(H_all[cur:cur+n]); cur += n
        return H_all, H_per_sample

    def score_items(self, H: torch.Tensor) -> torch.Tensor:
        target_dtype = self.score_head.net[0].weight.dtype
        return self.score_head(H.to(target_dtype))  # [B*K]

    def build_state_embed(self, H_sample: torch.Tensor, K_eval: int) -> torch.Tensor:
        """
        用于序列级 value：用候选平均 + 归一化的K_eval来表示 query-level 状态
        """
        pool = H_sample.mean(dim=0)
        remain_ratio = torch.tensor([K_eval / max(1, H_sample.size(0))],
                                    dtype=H_sample.dtype, device=H_sample.device)
        return torch.cat([pool, remain_ratio], dim=0)  # [D+1]

    def value(self, s_embed: torch.Tensor) -> torch.Tensor:
        return self.value_head(s_embed)


# =========================
# 全局 NDCG & PL：采样整条排名、计算 logprob
# =========================
def ndcg_at_k_from_perm(labels_vec: torch.Tensor, perm: List[int], K_eval: int) -> float:
    K_eval = min(K_eval, len(perm), labels_vec.size(0))
    rel = labels_vec[perm]
    gains = (2 ** rel[:K_eval] - 1).float()
    discounts = torch.log2(torch.arange(2, 2 + K_eval, device=rel.device).float())
    dcg = (gains / discounts).sum()
    rel_sorted = labels_vec.sort(descending=True).values
    idcg = ((2 ** rel_sorted[:K_eval] - 1).float() / discounts).sum()
    return float((dcg / (idcg + 1e-8)).item())


def pl_sample_permutation(scores_vec: torch.Tensor, K_eval: int) -> Tuple[List[int], torch.Tensor, List[torch.Tensor]]:
    """
    从 PL 分布采样一条长度 K_eval 的排名，返回：
      - perm: 绝对索引列表
      - logprob_sum_old: rollout 时旧策略的 log π_old(r|q)
      - ent_terms_old: 每步熵（旧策略），仅用于日志/可选熵奖励
    """
    remain = list(range(scores_vec.size(0)))
    perm: List[int] = []
    logp_terms = []
    ent_terms = []
    for _k in range(1, K_eval + 1):
        logits = scores_vec[remain].to(torch.float32)
        dist = Categorical(logits=logits)
        a_rel = dist.sample()
        logp_terms.append(dist.log_prob(a_rel))
        ent_terms.append(dist.entropy())
        perm.append(remain[a_rel.item()])
        remain.pop(a_rel.item())
    return perm, torch.stack(logp_terms).sum(), ent_terms


def pl_logprob_of_permutation(scores_vec: torch.Tensor, perm: List[int]) -> torch.Tensor:
    """对固定 perm 计算 log πθ(r|q)（用于新策略，建图回传）。"""
    remain = list(range(scores_vec.size(0)))
    terms = []
    for idx_abs in perm:
        j = remain.index(idx_abs)
        logits = scores_vec[remain].to(torch.float32)
        dist = Categorical(logits=logits)
        terms.append(dist.log_prob(torch.tensor(j, device=logits.device)))
        remain.pop(j)
    return torch.stack(terms).sum()


def pl_entropy_along_perm(scores_vec: torch.Tensor, perm: List[int]) -> torch.Tensor:
    """沿着 perm 的每一步计算当前策略的熵并求和（用于熵奖励）。"""
    remain = list(range(scores_vec.size(0)))
    ents = []
    for idx_abs in perm:
        logits = scores_vec[remain].to(torch.float32)
        dist = Categorical(logits=logits)
        ents.append(dist.entropy())
        j = remain.index(idx_abs)
        remain.pop(j)
    return torch.stack(ents).sum()


# =========================
# Rollout 记录（按样本一条序列）
# =========================
@dataclass
class SeqRec:
    sample_idx: int
    perm: List[int]
    logp_old_sum: torch.Tensor   # 旧策略 log π_old(r|q)
    reward: float                # 全局 NDCG@K_eval
    V_old: float                 # 旧策略的 V(s)（无梯度）
    ent_old_sum: float           # 旧策略的 per-step 熵之和（日志用）


# =========================
# 训练器：PPO-clip（序列级）
# =========================
class PLGlobalPPOTrainer:
    def __init__(self, pv: PolicyValue,
                 lr_actor: float = 3e-5, lr_critic: float = 1e-4, llm_lr_scale: float = 0.2,
                 eps_clip: float = 0.2, vf_coef: float = 0.5, ent_coef: float = 0.01,
                 max_grad_norm: float = 1.0,
                 K_eval: int = 10,
                 logit_temperature: float = 1.0,
                 adam8bit: bool = False):
        self.m = pv
        self.K_eval = K_eval
        self.eps = eps_clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_gn = max_grad_norm
        self.logit_temperature = logit_temperature

        llm_params   = list(self.m.llm.parameters())
        head_params  = list(self.m.score_head.parameters())
        critic_params= list(self.m.value_head.parameters())
        llm_lr = lr_actor * llm_lr_scale

        if adam8bit:
            import bitsandbytes as bnb
            self.opt_actor = bnb.optim.AdamW8bit(
                [{"params": llm_params, "lr": llm_lr},
                 {"params": head_params, "lr": lr_actor}],
                betas=(0.9, 0.95),
            )
            self.opt_critic = bnb.optim.AdamW8bit(
                critic_params, lr=lr_critic, betas=(0.9, 0.95)
            )
        else:
            self.opt_actor = torch.optim.AdamW(
                [{"params": llm_params, "lr": llm_lr},
                 {"params": head_params, "lr": lr_actor}],
                betas=(0.9, 0.95),
            )
            self.opt_critic = torch.optim.AdamW(
                critic_params, lr=lr_critic, betas=(0.9, 0.95)
            )

    @torch.no_grad()
    def rollout_batch_seq(self, batch: PackedBatch) -> List[SeqRec]:
        """
        对每个样本生成：
          - perm（整条序列）
          - logp_old_sum（旧策略）
          - reward = NDCG@K_eval
          - V_old（作为 baseline）
          - ent_old_sum（日志）
        """
        # 处理 DDP 包装的模型
        m = self.m.module if isinstance(self.m, DDP) else self.m
        dev = next(self.m.parameters()).device
        input_ids = batch.input_ids.to(dev)
        attention_mask = batch.attention_mask.to(dev)

        H_all, H_per_sample = m.encode_items_batched(input_ids, attention_mask,
                                                      batch.item_offsets, batch.pad_left)
        scores_all = m.score_items(H_all)
        if self.logit_temperature != 1.0:
            scores_all = scores_all / float(self.logit_temperature)

        B, K = batch.labels.size()
        K_eval = min(self.K_eval, K)
        seqs: List[SeqRec] = []
        for b in range(B):
            st, ed = b*K, (b+1)*K
            scores = scores_all[st:ed]
            perm, logp_sum, ent_terms = pl_sample_permutation(scores, K_eval=K_eval)
            R = ndcg_at_k_from_perm(batch.labels[b].to(dev), perm, K_eval=K_eval)

            # query 级状态嵌入（用 H_per_sample[b] 与 K_eval 构造）
            s_embed = m.build_state_embed(H_per_sample[b], K_eval=K_eval)
            V_old = float(m.value(s_embed).item())
            ent_old_sum = float(torch.stack(ent_terms).sum().item())

            seqs.append(SeqRec(sample_idx=b, perm=perm, logp_old_sum=logp_sum,
                               reward=R, V_old=V_old, ent_old_sum=ent_old_sum))
        return seqs

    def update_from_rollout_seq(self, batch: PackedBatch, seqs: List[SeqRec],
                                accum_steps: int = 1) -> Dict[str, float]:
        """
        PPO-clip（序列级）：
        L = -E[min(ratio*A, clip(ratio)*A)] + vf_coef*MSE(V, R) - ent_coef*H
        其中：
        - ratio = exp(logp_new_sum - logp_old_sum)
        - A = (R - V_old)  (旧价值作 baseline；单步情形无需 GAE)
        - H = per-step 熵之和（新策略、沿 perm）
        """
        # 处理 DDP 包装的模型
        m = self.m.module if isinstance(self.m, DDP) else self.m
        device = next(m.score_head.parameters()).device

        # 重新前向（建图）
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        H_all, H_per_sample = m.encode_items_batched(input_ids, attention_mask,
                                                      batch.item_offsets, batch.pad_left)
        scores_all = m.score_items(H_all)
        if self.logit_temperature != 1.0:
            scores_all = scores_all / float(self.logit_temperature)

        B, K = batch.labels.size()
        K_eval = min(self.K_eval, K)

        policy_losses, entropy_terms, value_losses = [], [], []

        # 日志
        rewards = []
        advs = []
        ratios = []

        for s in seqs:
            st, ed = s.sample_idx*K, (s.sample_idx+1)*K
            # —— 用 float32 做分布相关的计算（稳定）——
            scores = scores_all[st:ed].to(torch.float32)

            # 新策略 log πθ(r|q) ：沿 perm 重走一遍（float32）
            logp_new_sum = pl_logprob_of_permutation(scores, s.perm)  # returns float32
            ratio = torch.exp(logp_new_sum - s.logp_old_sum.detach()) # float32

            # Advantage（float32）
            A = torch.tensor(s.reward, device=device, dtype=torch.float32) - \
                torch.tensor(s.V_old,  device=device, dtype=torch.float32)

            # PPO-clip（float32）
            surr1 = ratio * A
            surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * A
            pol_loss = -torch.min(surr1, surr2)  # float32

            # 熵正则（float32）
            ent_sum_new = pl_entropy_along_perm(scores, s.perm)  # float32

            # Value：把 V_new 转成 float32 再和 float32 的 R 做 MSE
            s_embed = m.build_state_embed(H_per_sample[s.sample_idx], K_eval=K_eval)
            V_new = m.value(s_embed)                    # bf16
            V_new_f = V_new.float()                     # float32
            target_R = torch.tensor(s.reward, device=device, dtype=torch.float32)
            v_loss = F.mse_loss(V_new_f.squeeze(-1), target_R)  # float32

            policy_losses.append(pol_loss)
            entropy_terms.append(ent_sum_new)
            value_losses.append(v_loss)

            # 日志
            rewards.append(s.reward)
            advs.append(float(A.item()))
            ratios.append(float(ratio.detach().item()))

        policy_loss = torch.stack(policy_losses).mean().float()
        entropy_term = torch.stack(entropy_terms).mean().float()
        value_loss  = torch.stack(value_losses).mean().float()

        # 总损失（float32）
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_term

        (loss / max(1, accum_steps)).backward()

        logs = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_term.item()),
            "avg_reward_ndcg": float(sum(rewards) / max(1, len(rewards))),
            "avg_advantage": float(sum(advs) / max(1, len(advs))),
            "avg_ratio": float(sum(ratios) / max(1, len(ratios))),
        }
        return logs

    def clip_and_step(self):
        # 处理 DDP 包装的模型
        m = self.m.module if isinstance(self.m, DDP) else self.m
        torch.nn.utils.clip_grad_norm_(
            list(m.llm.parameters()) + list(m.score_head.parameters()),
            self.max_gn
        )
        torch.nn.utils.clip_grad_norm_(m.value_head.parameters(), self.max_gn)
        self.opt_actor.step(); self.opt_actor.zero_grad(set_to_none=True)
        self.opt_critic.step(); self.opt_critic.zero_grad(set_to_none=True)


# =========================
# Checkpoint（保存/恢复 LLM+Tokenizer+Heads+Opt+RNG）
# =========================
def _torch_rng_state():
    return {
        "python_random": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

def _torch_rng_load(state):
    random.setstate(state["python_random"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])

def save_checkpoint(output_dir: str,
                    step_global: int,
                    epoch_idx: int,
                    pv: PolicyValue,
                    opt_actor: torch.optim.Optimizer,
                    opt_critic: torch.optim.Optimizer,
                    tok,
                    args_dict: dict):
    ckpt_dir = Path(output_dir) / f"checkpoint-{step_global:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pv.llm.save_pretrained(str(ckpt_dir / "llm"))
    tok.save_pretrained(str(ckpt_dir / "tokenizer"))
    torch.save(pv.score_head.state_dict(), ckpt_dir / "score_head.pt")
    torch.save(pv.value_head.state_dict(), ckpt_dir / "value_head.pt")
    torch.save(opt_actor.state_dict(), ckpt_dir / "opt_actor.pt")
    torch.save(opt_critic.state_dict(), ckpt_dir / "opt_critic.pt")
    trainer_state = {"step_global": step_global, "epoch_idx": epoch_idx, "args": args_dict}
    with open(ckpt_dir / "trainer_state.json", "w") as f:
        json.dump(trainer_state, f, indent=2)
    torch.save(_torch_rng_state(), ckpt_dir / "rng_state.pt")
    print(f"[checkpoint] saved to {ckpt_dir}")

def _latest_checkpoint(output_dir: str) -> Optional[str]:
    patt = str(Path(output_dir) / "checkpoint-*")
    cks = sorted(glob.glob(patt))
    return cks[-1] if cks else None

def load_checkpoint(ckpt_dir: str,
                    pv: PolicyValue,
                    opt_actor: Optional[torch.optim.Optimizer],
                    opt_critic: Optional[torch.optim.Optimizer],
                    tok,
                    strict_opt: bool = False):
    ckpt_dir = Path(ckpt_dir)
    assert ckpt_dir.exists(), f"checkpoint not found: {ckpt_dir}"

    # 1) LLM + tokenizer + heads
    llm_loaded = Qwen2RRecCasualLM.from_pretrained(str(ckpt_dir / "llm"))
    pv.llm.load_state_dict(llm_loaded.state_dict())
    from transformers import AutoTokenizer as _Tok
    tok_loaded = _Tok.from_pretrained(str(ckpt_dir / "tokenizer"))
    tok.pad_token = tok_loaded.pad_token
    tok.pad_token_id = tok_loaded.pad_token_id
    tok.eos_token = tok_loaded.eos_token
    tok.eos_token_id = tok_loaded.eos_token_id
    tok.padding_side = "left"

    pv.score_head.load_state_dict(torch.load(ckpt_dir / "score_head.pt", map_location="cpu"))
    pv.value_head.load_state_dict(torch.load(ckpt_dir / "value_head.pt", map_location="cpu"))

    # 2) 优化器（可选）
    opt_actor_path = ckpt_dir / "opt_actor.pt"
    opt_critic_path = ckpt_dir / "opt_critic.pt"
    if opt_actor is not None:
        if opt_actor_path.exists():
            opt_actor.load_state_dict(torch.load(opt_actor_path, map_location="cpu"))
        elif strict_opt:
            raise FileNotFoundError(f"Missing {opt_actor_path}")
    if opt_critic is not None:
        if opt_critic_path.exists():
            opt_critic.load_state_dict(torch.load(opt_critic_path, map_location="cpu"))
        elif strict_opt:
            raise FileNotFoundError(f"Missing {opt_critic_path}")

    # 3) 训练状态（步数等，不影响评测）
    trainer_state_path = ckpt_dir / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path, "r") as f:
            trainer_state = json.load(f)
        step_global = int(trainer_state.get("step_global", 0))
        epoch_idx   = int(trainer_state.get("epoch_idx", 0))
    else:
        step_global, epoch_idx = 0, 0

    rng_path = ckpt_dir / "rng_state.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")
        _torch_rng_load(rng_state)

    print(f"[checkpoint] resumed from {ckpt_dir} (step={step_global}, epoch={epoch_idx})")
    return step_global, epoch_idx


# =========================
# Dataloader 构建(支持分布式)
# =========================
def build_dataloader_from_hf(dataset_path: str, split: str, tokenizer, prompter: BinaryItemPrompter,
                             batch_size: int, max_len_per_item: int,
                             seed: int, user_win_size: int, K: int, 
                             dataset_type: str = "movies",
                             is_distributed: bool = False, rank: int = 0, world_size: int = 1) -> DataLoader:
    ds_dict = hfds.load_from_disk(dataset_path)
    assert "item_info" in ds_dict, f"'item_info' split not found in {dataset_path}"
    assert split in ds_dict, f"'{split}' split not found. Available: {list(ds_dict.keys())}"
    item_info = ds_dict["item_info"]
    split_ds = ds_dict[split]

    rec_ds = RecFromHFUserDataset(
        split_ds=split_ds,
        item_info=item_info,
        K=K,
        user_win_size=user_win_size,
        seed=seed,
        dataset_type=dataset_type,
    )
    collate = PackedCollate(tokenizer, prompter, max_len_per_item=max_len_per_item, K=K)
    
    # 使用固定种子的生成器确保结果可复现（训练和测试都使用相同逻辑）
    g = torch.Generator()
    g.manual_seed(seed)
    
    if is_distributed:
        sampler = DistributedSampler(rec_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
        return DataLoader(rec_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate, drop_last=True)
    else:
        return DataLoader(rec_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True, generator=g)


# =========================
# 主函数
# =========================
def parse_args():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--dataset_path", type=str, required=True, help="HF datasets.load_from_disk directory")
    ap.add_argument("--dataset_type", type=str, default="movies", choices=["movies", "other"], 
                    help="Dataset type: 'movies' for movie recommendations with enhanced features, 'other' for general items")
    ap.add_argument("--split", type=str, default="train", choices=["train", "validation", "test", "valid"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr_actor", type=float, default=3e-5)
    ap.add_argument("--lr_critic", type=float, default=1e-4)
    ap.add_argument("--llm_lr_scale", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--eps_clip", type=float, default=0.2)
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--user_window_size", type=int, default=10)
    ap.add_argument("--max_len_per_item", type=int, default=512)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--K_eval", type=int, default=10, help="采样/评测的 top-K（<=K）")
    ap.add_argument("--freeze_llm", action="store_true")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--adam8bit", action="store_true")
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--flash_attn", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--pair_micro_bs", type=int, default=64)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    # ckpt
    ap.add_argument("--output_dir", type=str, default="./outputs")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--resume_from", type=str, default=None)  # "last" or path
    # 探索温度
    ap.add_argument("--logit_temperature", type=float, default=1.0)
    ap.add_argument("--preview_n", type=int, default=1, help="每次打印的样本条数（每样本随机挑1个候选 prompt）")
    ap.add_argument("--preview_max_new_tokens", type=int, default=256, help="预览生成的最大新 token 数")
    ap.add_argument("--token_log_path", type=str, default=None, help="可选：把预览的 token 长度统计追加写入到该 JSONL 文件")
    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # 单机多卡场景
        rank = 0
        world_size = torch.cuda.device_count()
        local_rank = 0
        
        if world_size > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    global BINARY_PROMPT_TMPL
    
    args = parse_args()
    
    # 根据数据集类型设置 Prompt 模板
    if args.dataset_type == "movies":
        BINARY_PROMPT_TMPL = BINARY_PROMPT_TMPL_MOVIES
    else:
        BINARY_PROMPT_TMPL = BINARY_PROMPT_TMPL_OTHER
    
    # 初始化分布式训练
    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    
    # 设置设备
    if is_distributed:
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 只在主进程打印信息
    if is_main_process:
        if is_distributed:
            print(f"=== Distributed Training: {world_size} GPUs ===")
            print(f"Rank {rank}/{world_size}, Local rank: {local_rank}")
        else:
            print("=== Single GPU Training ===")
        print(f"Dataset type: {args.dataset_type}")
    
    set_seed(args.seed + rank)  # 每个进程使用不同的随机种子

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "<|pad|>"

    prompter = BinaryItemPrompter(tok, emb_token=getattr(tok, "generation_end", "<answer>"),
                                  emb_end_token="</answer>", dataset_type=args.dataset_type)
    split = "valid" if args.split == "validation" else args.split
    dl = build_dataloader_from_hf(
        dataset_path=args.dataset_path,
        split=split,
        tokenizer=tok,
        prompter=prompter,
        batch_size=args.batch_size,
        max_len_per_item=args.max_len_per_item,
        seed=args.seed,
        user_win_size=args.user_window_size,
        K=args.K,
        dataset_type=args.dataset_type,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )

    # LLM
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    llm = Qwen2RRecCasualLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=("flash_attention_2" if args.flash_attn else None),
    )
    if hasattr(llm, "config"):
        llm.config.use_cache = False
    if args.grad_ckpt and hasattr(llm, "gradient_checkpointing_enable"):
        try:
            llm.gradient_checkpointing_enable()
        except Exception:
            pass

    D = llm.config.hidden_size
    score_head = ScoreHead(D).to(dtype=next(llm.parameters()).dtype)
    value_head = ValueHead(D + 1).to(dtype=next(llm.parameters()).dtype)

    pv = PolicyValue(
        llm=llm,
        score_head=score_head,
        value_head=value_head,
        pair_micro_bs=args.pair_micro_bs,
        pad_token_id=tok.pad_token_id,
        freeze_llm=args.freeze_llm
    ).to(device)

    # pad/eos 对齐 (在 DDP 包装之前)
    if getattr(pv.llm, "config", None) is not None:
        if getattr(pv.llm.config, "pad_token_id", None) is None:
            pv.llm.config.pad_token_id = tok.pad_token_id
        if getattr(pv.llm.config, "eos_token_id", None) is None and tok.eos_token_id is not None:
            pv.llm.config.eos_token_id = tok.eos_token_id
    if getattr(pv.llm, "generation_config", None) is not None:
        if pv.llm.generation_config.pad_token_id is None:
            pv.llm.generation_config.pad_token_id = tok.pad_token_id
        if pv.llm.generation_config.eos_token_id is None and tok.eos_token_id is not None:
            pv.llm.generation_config.eos_token_id = tok.eos_token_id
    if getattr(pv.llm, "pad_token_id", None) is None:
        pv.llm.pad_token_id = tok.pad_token_id

    # 在 DDP 包装之前创建 Trainer (这样 Trainer 可以访问原始模型)
    trainer = PLGlobalPPOTrainer(
        pv,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        llm_lr_scale=args.llm_lr_scale,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        K_eval=args.K_eval,
        logit_temperature=args.logit_temperature,
        adam8bit=args.adam8bit,
    )
    
    # 使用 DDP 包装模型 (在创建 Trainer 之后)
    if is_distributed:
        pv = DDP(pv, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        # 更新 trainer 中的模型引用为 DDP 包装后的模型
        trainer.m = pv
        # 获取原始模型的引用(用于保存检查点)
        pv_module = pv.module
    else:
        pv_module = pv

    # ckpt 恢复
    step_global = 0
    start_epoch = 0
    if is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.resume_from is not None:
        resume_dir = _latest_checkpoint(args.output_dir) if args.resume_from == "last" else args.resume_from
        if resume_dir is not None:
            step_global, start_epoch = load_checkpoint(resume_dir, pv_module, trainer.opt_actor, trainer.opt_critic, tok)
    
    # 同步所有进程
    if is_distributed:
        dist.barrier()

    accum = max(1, args.grad_accum_steps)
    trainer.opt_actor.zero_grad(set_to_none=True)
    trainer.opt_critic.zero_grad(set_to_none=True)
    metrics_history: List[Dict[str, Any]] = []

    for ep in range(start_epoch, args.epochs):
        # 设置分布式采样器的 epoch (确保每个 epoch 的数据打乱不同)
        if is_distributed and hasattr(dl, 'sampler') and hasattr(dl.sampler, 'set_epoch'):
            dl.sampler.set_epoch(ep)
        
        # 只在主进程显示进度条
        iterator = tqdm(dl, desc=f"Epoch {ep+1}/{args.epochs}") if is_main_process else dl
        
        for i, packed in enumerate(iterator, start=1):
            packed.input_ids = packed.input_ids.to(device)
            packed.attention_mask = packed.attention_mask.to(device)
            packed.labels = packed.labels.to(device)

            # 1) rollout（无梯度）
            seqs = trainer.rollout_batch_seq(packed)

            # 2) update（建图与反传）
            logs = trainer.update_from_rollout_seq(packed, seqs, accum_steps=accum)
            step_global += 1
            
            if is_main_process:
                metrics_history.append({"step": step_global, "epoch": ep, **logs})

            if (i % accum) == 0:
                trainer.clip_and_step()

            # 只在主进程打印
            if is_main_process and step_global % args.print_every == 0:
                print(f"[ep {ep} step {step_global}] "
                    f"loss={logs['loss']:.4f} pol={logs['policy_loss']:.4f} "
                    f"val={logs['value_loss']:.4f} ent={logs['entropy']:.4f} "
                    f"avg_ndcg={logs['avg_reward_ndcg']:.4f} "
                    f"adv={logs['avg_advantage']:.4f} ratio={logs['avg_ratio']:.4f}")
                # ★ 打印 LLM 实际生成文本（用当前模型）
                preview_answers(
                    pv_module.llm, tok, prompter, 
                    packed.raw_prompts_20, 
                    preview_n=args.preview_n, 
                    max_new_tokens=args.preview_max_new_tokens,
                    max_len_per_item=args.max_len_per_item,  # 使用与训练相同的截断长度
                    log_path=args.token_log_path,
                    step=step_global
                )
                

            # 只在主进程保存检查点
            if is_main_process and step_global % args.save_every == 0:
                if (i % accum) != 0:
                    trainer.clip_and_step()
                save_checkpoint(
                    output_dir=args.output_dir,
                    step_global=step_global,
                    epoch_idx=ep,
                    pv=pv_module,
                    opt_actor=trainer.opt_actor,
                    opt_critic=trainer.opt_critic,
                    tok=tok,
                    args_dict=vars(args)
                )
            
            # 同步所有进程 (在保存检查点后)
            if is_distributed and step_global % args.save_every == 0:
                dist.barrier()

    if (step_global % accum) != 0:
        trainer.clip_and_step()

    if is_main_process and metrics_history:
        curve_path = Path(args.output_dir) / "training_curve.jsonl"
        with open(curve_path, "w") as f:
            for record in metrics_history:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        print(f"Training curve saved to {curve_path}")

    if is_main_process:
        print("Training finished.")
    
    # 清理分布式训练环境
    cleanup_distributed()


if __name__ == "__main__":
    main()