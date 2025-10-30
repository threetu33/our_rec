#!/bin/bash

# 完全修复版本的R1兼容测试脚本
# 严格按照训练时的评估流程进行推理

set -e

# 默认参数
CHECKPOINT_PATH="/data/hongdeyao/code/RRec/RRec_checkpoints/checkpoints/run_name/checkpoint-6058"
DATASET_DIR="/data/hongdeyao/Musical_Instruments_0_2022-10-2023-10"
TARGET_SAMPLES=200
MODEL_TYPE="qwen"
DEVICE="cuda:2"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --dataset_dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --target_samples)
            TARGET_SAMPLES="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --checkpoint_path PATH    Checkpoint path (default: $CHECKPOINT_PATH)"
            echo "  --dataset_dir PATH        Dataset directory (default: $DATASET_DIR)"
            echo "  --target_samples NUM      Number of samples (default: $TARGET_SAMPLES)"
            echo "  --model_type TYPE         Model type (default: $MODEL_TYPE)"
            echo "  --device DEVICE           Device to use (default: $DEVICE)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🚀 Training-Consistent Checkpoint Test (V2)"
echo "============================================"
echo "📁 Checkpoint: $CHECKPOINT_PATH"
echo "📊 Dataset: $DATASET_DIR"
echo "🎯 Target samples: $TARGET_SAMPLES"
echo "🤖 Model: $MODEL_TYPE"
echo "💻 Device: $DEVICE"
echo "============================================"
echo "✅ This version strictly follows training evaluation process"
echo "✅ Uses correct prompters and reasoning generation"
echo "✅ Saves complete reasoning (not truncated)"
echo "✅ R1-compatible candidate sampling (100 items)"
echo "============================================"

# 检查路径
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint path not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Dataset directory not found: $DATASET_DIR"
    exit 1
fi

# 检查Python脚本是否存在
PYTHON_SCRIPT="/data/hongdeyao/code/RRec/run_rrec.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# 设置环境变量
# export CUDA_VISIBLE_DEVICES=${DEVICE#cuda:}
export TOKENIZERS_PARALLELISM=false

echo ""
echo "🔄 Starting training-consistent evaluation..."
echo "============================================="

# 运行Python脚本
python3 "$PYTHON_SCRIPT" \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --dataset_dir="$DATASET_DIR" \
    --num_samples="$TARGET_SAMPLES" \
    --model_type="$MODEL_TYPE" \
    --device="$DEVICE" \
    --seed=42 \
    --reference_json="/data/hongdeyao/code/RRec/sft/tmp/outputs/checkpoint-022000_win5_2_w1p00_0p00_minhist15.json" \
    --user_window_size=5

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 TRAINING-CONSISTENT EVALUATION COMPLETED!"
    echo "==========================================="
    echo "📁 Checkpoint: $CHECKPOINT_PATH"
    echo "🎯 Samples tested: $TARGET_SAMPLES"
    echo "🔍 Each sample had ~100 candidates (same as run_r1.py)"
    echo "✅ Evaluation process matches training exactly"
    echo "✅ Complete reasoning saved (not truncated)"
    echo "✅ Candidate items saved for R1 comparison"
    echo ""
    echo "📋 Check the generated JSON file for detailed results"
    echo "📋 File name: training_consistent_results_${MODEL_TYPE}_${TARGET_SAMPLES}samples_*.json"
else
    echo "❌ Evaluation failed"
    exit 1
fi

echo "✅ Training-consistent checkpoint test completed successfully!"
