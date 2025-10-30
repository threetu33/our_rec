# eval_pl_global_ppo.py
# Evaluate PL-ranking policy on HF dataset test split (one-by-one).
# Modes: rollout (PL-sampled) / greedy (argmax-by-scores).
from __future__ import annotations
import argparse
from pathlib import Path
import random
import torch
from tqdm import tqdm
from transformers import GenerationConfig
import json

# ==== import from your training file ====
from train import (
    Qwen2RRecCasualLM, AutoTokenizer,
    BinaryItemPrompter, build_dataloader_from_hf,
    PolicyValue, ScoreHead, ValueHead,
    ndcg_at_k_from_perm, pl_sample_permutation, _latest_checkpoint, load_checkpoint,
    extract_between  # 若不在 train 里暴露，就把该函数复制过来
)

@torch.no_grad()
def eval_one_sample(pv: PolicyValue,
                    batch,
                    K_eval: int,
                    mode: str,
                    logit_temperature: float):
    """
    对单个样本（batch_size=1）计算一次 NDCG@K_eval，并返回：
    ndcg(float), scores(torch.Tensor[K]), perm(list[int]), candidate_ids(list)
    """
    device = next(pv.parameters()).device
    input_ids = batch.input_ids.to(device)      # [1, L]
    attention_mask = batch.attention_mask.to(device)  # [1, L]
    labels = batch.labels.to(device)            # [1, K]
    H_all, H_per_sample = pv.encode_items_batched(input_ids, attention_mask,
                                                  batch.item_offsets, batch.pad_left)
    scores_all = pv.score_items(H_all)          # [1*K]
    if logit_temperature != 1.0:
        scores_all = scores_all / float(logit_temperature)

    K = labels.size(1)
    K_eval = min(K_eval, K)
    scores = scores_all.view(K).float()         # [K]

    if mode == "rollout":
        perm, _, _ = pl_sample_permutation(scores, K_eval=K_eval)
    elif mode == "greedy":
        perm = torch.argsort(scores, descending=True)[:K_eval].tolist()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ndcg = ndcg_at_k_from_perm(labels[0], perm, K_eval=K_eval)
    
    # 提取候选item ID（如果batch中包含该信息）
    candidate_ids = getattr(batch, 'candidate_ids', [None] * K)
    if candidate_ids and len(candidate_ids) > 0:
        candidate_ids = candidate_ids[0] if isinstance(candidate_ids[0], list) else candidate_ids
    
    return float(ndcg), scores, perm, candidate_ids


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--dataset_path", type=str, required=True,
                    help="HF datasets.load_from_disk directory (must contain 'test' and 'item_info')")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--K_eval", type=int, default=10)
    ap.add_argument("--max_len_per_item", type=int, default=512)
    ap.add_argument("--user_window_size", type=int, default=10)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--flash_attn", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mode", type=str, default="rollout", choices=["rollout", "greedy"])
    ap.add_argument("--logit_temperature", type=float, default=1.0)
    ap.add_argument("--num_samples", type=int, default=200,
                    help="评测多少个样本（-1 表示全量）")
    ap.add_argument("--resume_from", type=str, default=None,
                    help='"last" to auto-pick latest checkpoint in --output_dir, or a path to checkpoint dir')
    ap.add_argument("--output_dir", type=str, default="./outputs",
                    help="Used only when --resume_from=last")
    ap.add_argument("--dataset_type", type=str, default="other", choices=["movies", "other"], 
                    help="Dataset type to determine which prompt template and text formatting to use")

    # 可视化生成设置
    ap.add_argument("--print_every_text", type=int, default=20,
                    help="每隔多少个样本，打印一次模型生成文本（<=0 关闭）")
    ap.add_argument("--gen_mode", type=str, default="top1", choices=["top1", "random", "all"],
                    help="选择用 Top-1 候选、随机候选或所有候选来做生成展示")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 根据数据集类型设置prompt模板
    from train import BINARY_PROMPT_TMPL_MOVIES, BINARY_PROMPT_TMPL_OTHER
    import train
    if args.dataset_type == "movies":
        train.BINARY_PROMPT_TMPL = BINARY_PROMPT_TMPL_MOVIES
        print(f"Using MOVIES prompt template for dataset type: {args.dataset_type}")
    else:
        train.BINARY_PROMPT_TMPL = BINARY_PROMPT_TMPL_OTHER
        print(f"Using OTHER prompt template for dataset type: {args.dataset_type}")

    # Tokenizer & prompter
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "<|pad|>"

    prompter = BinaryItemPrompter(tok, emb_token=getattr(tok, "generation_end", "<answer>"),
                                  emb_end_token="</answer>", dataset_type=args.dataset_type)

    # DataLoader: 一次只取 1 个样本
    test_loader = build_dataloader_from_hf(
        dataset_path=args.dataset_path,
        split="test",
        tokenizer=tok,
        prompter=prompter,
        batch_size=1,  # 关键：逐个样本
        max_len_per_item=args.max_len_per_item,
        seed=42,
        user_win_size=args.user_window_size,
        K=args.K,
        dataset_type=args.dataset_type,
    )

    # Model
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    llm = Qwen2RRecCasualLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=("flash_attention_2" if args.flash_attn else None),
    )
    if hasattr(llm, "config"):
        llm.config.use_cache = False

    D = llm.config.hidden_size
    score_head = ScoreHead(D).to(dtype=next(llm.parameters()).dtype)
    value_head = ValueHead(D + 1).to(dtype=next(llm.parameters()).dtype)

    pv = PolicyValue(
        llm=llm,
        score_head=score_head,
        value_head=value_head,
        pair_micro_bs=64,
        pad_token_id=tok.pad_token_id,
        freeze_llm=False,
    ).to(device)

    # Align pad/eos
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

    # 可选：加载 checkpoint
    if args.resume_from is not None:
        ckpt_dir = _latest_checkpoint(args.output_dir) if args.resume_from == "last" else args.resume_from
        if ckpt_dir is not None:
            load_checkpoint(ckpt_dir, pv, opt_actor=None, opt_critic=None, tok=tok)

    pv.eval()

    total_ndcg = 0.0
    n_done = 0
    n_skipped = 0  # 跳过的样本数
    detailed_results = []  # 用于保存详细结果
    pbar = tqdm(test_loader, desc=f"[TEST-1by1] mode={args.mode} K_eval={args.K_eval}", unit="sample")

    # 生成配置
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True, temperature=args.temperature, top_p=args.top_p,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )

    for batch in pbar:
        # 检查用户交互历史长度是否符合条件（15-20之间）
        raw_sample = batch.raw_samples[0] if hasattr(batch, 'raw_samples') and len(batch.raw_samples) > 0 else {}
        history_item_ids = raw_sample.get('history_item_id', [])
        history_length = len(history_item_ids) if history_item_ids else 0
        
        # 如果历史长度不在15-20范围内，跳过这个样本
        if history_length < 0 or history_length > 20:
            n_skipped += 1
            pbar.set_postfix({"done": f"{n_done}", "skipped": f"{n_skipped}", "hist_len": f"{history_length}"})
            continue
        # 单样本评测
        ndcg, scores, perm, candidate_ids = eval_one_sample(
            pv=pv,
            batch=batch,
            K_eval=args.K_eval,
            mode=args.mode,
            logit_temperature=args.logit_temperature,
        )
        total_ndcg += ndcg
        n_done += 1
        running = total_ndcg / n_done
        pbar.set_postfix({"ndcg": f"{ndcg:.4f}", "avg": f"{running:.4f}", "done": f"{n_done}", "skipped": f"{n_skipped}", "hist_len": f"{history_length}"})
        
        # 收集详细结果信息
        labels = batch.labels[0].cpu().tolist() if hasattr(batch, 'labels') else []
        target_idx = labels.index(1) if 1 in labels else -1
        target_id = candidate_ids[target_idx] if target_idx >= 0 and target_idx < len(candidate_ids) else None
        
        # 获取原始样本信息
        sample_idx = batch.sample_indices[0] if hasattr(batch, 'sample_indices') and len(batch.sample_indices) > 0 else n_done - 1
        user_id = raw_sample.get('user_id') if raw_sample else None
        
        sample_result = {
            'sample_index': int(sample_idx) if sample_idx is not None else n_done - 1,
            'user_id': str(user_id) if user_id is not None else None,
            'history_length': history_length,  # 添加历史长度信息
            'ndcg': float(ndcg),
            'candidate_ids': [int(x) if x is not None else None for x in candidate_ids],
            'scores': scores.cpu().tolist(),
            'predicted_ranking': perm,
            'labels': labels,
            'target_item_id': int(target_id) if target_id is not None else None,
            'target_position_in_candidates': target_idx,
            'target_rank_in_prediction': perm.index(target_idx) + 1 if target_idx in perm else -1
        }
        detailed_results.append(sample_result)

        # 每隔 N 个样本，打印一次 LLM 生成
        if args.print_every_text > 0 and (n_done % args.print_every_text == 0):
            # 从该样本的 20 个 prompt 里选择：Top-1、随机或所有
            prompts_20 = batch.raw_prompts_20[0]  # List[str] for this sample
            
            if args.gen_mode == "top1":
                # 取当前样本分数最高的候选对应的 prompt
                top1 = int(torch.argmax(scores).item())
                prompt_texts = [prompts_20[top1]]
                prompt_indices = [top1]
            elif args.gen_mode == "random":
                idx = random.randint(0, len(prompts_20) - 1)
                prompt_texts = [prompts_20[idx]]
                prompt_indices = [idx]
            elif args.gen_mode == "all":
                # 打印所有20个候选的回答
                prompt_texts = prompts_20
                prompt_indices = list(range(len(prompts_20)))
            else:
                raise ValueError(f"Unknown gen_mode: {args.gen_mode}")

            print("\n" + "="*80)
            print(f"[sample #{n_done}] NDCG@{args.K_eval}={ndcg:.4f} | avg={running:.4f}")
            print(f"Generation mode: {args.gen_mode} | Total prompts to generate: {len(prompt_texts)}")
            print("="*80)

            # 对每个选中的 prompt 进行生成
            for i, (prompt_text, prompt_idx) in enumerate(zip(prompt_texts, prompt_indices)):
                # 真正生成
                inputs = tok(prompt_text, return_tensors="pt").to(device)
                out_ids = pv.llm.generate(**inputs, generation_config=gen_cfg)
                out_txt = tok.decode(out_ids[0], skip_special_tokens=True)
                # 抽取答案段（若你在 train 里有 extract_between）
                try:
                    ans = extract_between(out_txt, prompter.emb_token, prompter.emb_end_token)
                except Exception:
                    ans = out_txt

                print(f"\n--- Candidate #{prompt_idx+1} (score: {scores[prompt_idx]:.4f}) ---")
                if args.gen_mode == "all":
                    # all模式下显示更完整的内容
                    print("PROMPT (full):")
                    print(prompt_text)
                    print("\nRAW OUTPUT (full):")
                    print(out_txt)
                else:
                    # 其他模式保持原有的截断显示
                    print("PROMPT (trunc):", prompt_text[:220].replace("\n", " ") + " ...")
                    print("RAW OUTPUT:", out_txt[-300:] if len(out_txt) > 300 else out_txt)
                print("ANSWER between tags:", ans)
                
                if args.gen_mode == "all" and i < len(prompt_texts) - 1:
                    print("-" * 60)
            
            print("="*80 + "\n")

        if args.num_samples > 0 and n_done >= args.num_samples:
            break

    avg_ndcg = total_ndcg / max(1, n_done)
    
    print("=" * 60)
    print(f"Eval mode: {args.mode} | split: test | samples: {n_done} | skipped: {n_skipped}")
    print(f"History length filter: 15-20 | Total processed: {n_done + n_skipped}")
    print(f"Average NDCG@{args.K_eval}: {avg_ndcg:.6f}")
    print("=" * 60)
    
    # 保存详细结果到JSON，文件名与checkpoint目录名一致（如checkpoint-018500.json）
    ckpt_name = None
    if args.resume_from is not None:
        ckpt_dir = _latest_checkpoint(args.output_dir) if args.resume_from == "last" else args.resume_from
        if ckpt_dir is not None:
            ckpt_name = Path(ckpt_dir).name  # checkpoint-018500
    if ckpt_name:
        results_file = Path(args.output_dir) / f"{ckpt_name}.json"
    else:
        results_file = Path(args.output_dir) / f"test_results_{args.mode}_K{args.K_eval}.json"
    results_data = {
        'config': {
            'model_name_or_path': args.model_name_or_path,
            'dataset_path': args.dataset_path,
            'dataset_type': args.dataset_type,
            'mode': args.mode,
            'K': args.K,
            'K_eval': args.K_eval,
            'num_samples': n_done,
            'num_skipped': n_skipped,
            'history_length_filter': '15-20',
            'logit_temperature': args.logit_temperature,
            'resume_from': args.resume_from
        },
        'metrics': {
            'average_ndcg': avg_ndcg,
            'total_samples': n_done,
            'total_skipped': n_skipped,
            'total_processed': n_done + n_skipped,
            'history_length_stats': {
                'min_length': 15,
                'max_length': 20,
                'filter_applied': True
            }
        },
        'detailed_results': detailed_results
    }
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Detailed results saved to: {results_file}")
    print(f"   Total samples meeting criteria (history length 15-20): {len(detailed_results)}")
    print(f"   Total samples skipped (history length outside 15-20): {n_skipped}")
    print(f"   Total samples processed: {n_done + n_skipped}")


if __name__ == "__main__":
    main()