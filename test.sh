python test.py \
    --dataset_path /data/hongdeyao/Musical_Instruments_0_2022-10-2023-10 \
    --dataset_type other \
    --model_name_or_path /data/hongdeyao/code/RRec/sft/data/sft_check_output_full_num_3000_yes1000_no2000_concise/checkpoint-33 \
    --resume_from /data/hongdeyao/code/RRec/sft/rl_result_num_3000_yes1000_no2000_concise/checkpoint-022000 \
    --mode greedy \
    --output_dir ./outputs \
    --K 20 --K_eval 10 \
    --num_samples 8 \
    --print_every_text 8 \
    --gen_mode all \
    --max_new_tokens 512 \
    --max_len_per_item 1024 \
    --device cuda:0

# --model_name_or_path /data/public_checkpoints/huggingface_models/Qwen2.5-3B-Instruct \
# --resume_from /data/zhengkehan/rrec_rank_globalppo_A/checkpoint-066000 \