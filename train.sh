export CUDA_VISIBLE_DEVICES=7,3

# 自动检测可用的 GPU 数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "Using $NUM_GPUS GPUs: $CUDA_VISIBLE_DEVICES"

# 使用 torchrun 启动分布式训练
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 train.py \
  --dataset_path /data/hongdeyao/Video_Games_0_2022-10-2023-10 \
  --model_name_or_path /data/public_checkpoints/huggingface_models/Qwen2.5-3B-Instruct \
  --output_dir /data/hongdeyao/code/RRec/sft/rl_result_video_games \
  --dataset_type other \
  --split train \
  --epochs 3 \
  --batch_size 1 \
  --K 20 \
  --K_eval 10 \
  --max_len_per_item 450 \
  --preview_max_new_tokens 2000 \
  --user_window_size 10 \
  --seed 42 \
  --print_every 200 \
  --save_every 4000 \
  --dtype bf16 \
  --grad_ckpt \
  --grad_accum_steps 4 \
  --pair_micro_bs 1 \
  --adam8bit \
  --llm_lr_scale 0.2 \
  --ent_coef 0.01 \
  --vf_coef 0.5 \
  --eps_clip 0.2 \
  --logit_temperature 1.0 \
  --token_log_path /data/hongdeyao/code/RRec/sft/rl_result_num_1500_yes500_no1000_movies/token_lengths.jsonl
