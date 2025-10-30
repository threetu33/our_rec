# our rec

## 项目简介

基于强化学习（PPO）和大语言模型的推荐系统项目。支持使用Amazon Review数据集进行训练和评估。

## 目录结构概览

```
rrec/
├── train.py                    # PPO强化学习训练主程序
├── test.py                     # 模型测试评估主程序
├── preprocess.py               # 数据预处理脚本
├── paths.py                    # 模型路径配置
├── requirements.txt            # Python依赖
├── train.sh / test.sh          # 训练/测试快捷脚本
├── models/                     # 模型定义
├── prompters/                  # Prompt生成器
├── trainers/                   # 训练器
├── sft_train/                  # SFT训练相关
├── compare_model/              # 对比模型评估
└── tools/                      # 辅助工具脚本
```

## 安装依赖

`pip install -r requirements.txt`

若是中国用户，建议：
1. `pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1`
2. 注释掉 `torch==2.5.1+cu121`
3. `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`

## 训练流程

> 如果你不准备做SFT训练，请跳过步骤2，直接使用基础模型进行RL训练

1. 数据处理：下载 amazon 数据集并且切分为训练集、验证集、测试集
```bash
python preprocess.py \
      --category Musical_Instruments \
      --K 0 \
      --st_year 2022 \
      --st_month 10 \
      --ed_year 2023 \
      --ed_month 10 \
      --data_root_dir /path/to/your/dir
```

2. SFT 训练
- 准备 prompts
  ```bash
  python sft_train/generate_prompts.py \
        --dataset_dir /data2/hongdeyao/Movies_and_TV_0_2022-10-2023-10 \
        --num_samples 10000 \
        --match_ratio 0.5 \
        --model_type qwen \
        --output_file /path/to/your/output/file
  ```

- 为 prompts 生成 response
  ```bash
  python sft_train/get_response_by_deepseek.py
  ```
  你可能需要修改的参数（在代码中）：
  - `input_file`：刚刚准备的prompts文件地址
  - `output_file`：生成好response的文件保存地址
  - `prompt_id_max`：生成response的id上限
  - `prompt_id_min`：生成response的id下限

- 筛选正确的response并处理数据
  ```bash
  python sft_train/process_sft_data.py
  ```
  你可能需要修改的参数（在代码中）：
  - `input_file`：刚刚生成好response的文件地址
  - `output_file`：筛选好数据的文件保存地址

- sft训练流程(建议使用llama-factory)
  - 修改sft.yaml中的dataset和output_dir
  - 同步更新data_info.json中的字段
  - 把训练数据文件移动到data文件夹
  - `CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train sft.yaml`

3. RL 训练
```bash
bash train.sh
```
你可能需要修改的参数：
- `BINARY_PROMPT_TMPL_MOVIES(train.py)`：如果你使用的是SFT模型，请修改为你SFT模型的prompt
- `BINARY_PROMPT_TMPL_OTHER(train.py)`：如果你使用的是SFT模型，请修改为你SFT模型的prompt
- `CUDA_VISIBLE_DEVICES`：使用的gpu编号
- `dataset_path`：数据集位置
- `dataset_type`：movies或者other
- `model_name_or_path`：待训练的模型位置（基础模型或SFT模型）
- `output_dir`：训练完的模型保存地址
- `max_len_per_item`：每个item最多占用的token数
- `user_window_size`：user交互历史的数目上限
- `print_every`：打印日志的频率
- `save_every`：保存checkpoint的频率
- `token_log_path`：记录输出长度的文件地址

## 测试流程

1. **our_rec** (RL训练后的模型)
```bash
bash test.sh
```
你可能需要修改的参数：
- `BINARY_PROMPT_TMPL_MOVIES(train.py)`：如果你使用的是SFT模型，请修改为你SFT模型的prompt
- `BINARY_PROMPT_TMPL_OTHER(train.py)`：如果你使用的是SFT模型，请修改为你SFT模型的prompt
- `device`：使用的gpu
- `output_dir`：测试结果文件保存位置
- `dataset_path`：数据集位置
- `dataset_type`：movies或者other
- `model_name_or_path`：基础模型位置
- `resume_from`：训练完的checkpoint地址
- `num_samples`：测试的样本数目
- `gen_mode`：打印示例输出的方式(top1/random/all)
- `max_len_per_item`：每个item最多占用的token数
- `user_window_size`：user交互历史的数目上限
- `print_every_text`：打印示例输出的频率

2. **rrec** (使用与训练一致的评估流程)
```bash
python compare_model/rrec.py \
    --checkpoint_path /trained/rrec/model/checkpoint \
    --dataset_dir /data2/hongdeyao/Musical_Instruments_0_2022-10-2023-10 \
    --num_samples 200 \
    --model_type qwen \
    --device cuda:0 \
    --seed 42 \
    --reference_json /path/to/our_rec/output/json \
    --user_window_size 10
```

3. **deepseek** (使用DeepSeek API评估)
```bash
python compare_model/deepseek.py \
    --test_samples 200 \
    --dataset_dir /data2/hongdeyao/Musical_Instruments_0_2022-10-2023-10 \
    --test_only \
    --candidate_mode json \
    --candidate_json_path /path/to/our_rec/output/json
```

## 代码文件基本信息

### 数据处理

- **`preprocess.py`** - 数据预处理脚本
  - 从Amazon Review数据集下载并处理原始数据
  - 若发现item数目不足10000，会自动向前扩展3个月
  - 主要参数：
    - `--category` - 数据集类别（如Musical_Instruments, Video_Games等）
    - `--K` - 用户最少交互历史数量（最多设置为20，设置为0表示不限制）
    - `--st_year/st_month` - 数据起始年月
    - `--ed_year/ed_month` - 数据结束年月
    - `--window_size` - 历史窗口大小（默认10）
    - `--data_root_dir` - 输出目录
  - 运行示例：
    ```bash
    python preprocess.py \
      --category Musical_Instruments \
      --K 0 \
      --st_year 2022 \
      --st_month 10 \
      --ed_year 2023 \
      --ed_month 10 \
      --data_root_dir /data2/hongdeyao
    ```

- **`sft_train/generate_prompts.py`** - 生成二分类判断prompt
  - 从数据集随机抽取样本并生成用于判断物品是否匹配用户偏好的prompt
  - 注意：修改内部的`judgment_instruction`，`prompt`，`self.test_data`
  - 主要参数：
    - `--dataset_dir` - 数据集目录
    - `--num_samples` - 生成样本数量
    - `--match_ratio` - 匹配样本比例
    - `--model_type` - 模型类型
    - `--output_file` - 输出文件路径
  - 运行示例：
    ```bash
    python sft_train/generate_prompts.py \
      --dataset_dir /data2/hongdeyao/Movies_and_TV_0_2022-10-2023-10 \
      --num_samples 100 \
      --match_ratio 0.5 \
      --model_type qwen \
      --output_file /path/to/output/file
    ```

- **`sft_train/get_response_by_deepseek.py`** - 使用DeepSeek API生成response
  - 需修改代码中的路径：
    - `input_file` - 输入prompt文件路径
    - `output_file` - 输出结果文件路径
    - `prompt_id_max` - 生成response的id上限
    - `prompt_id_min` - 生成response的id下限
  - 运行示例：
    ```bash
    python sft_train/get_response_by_deepseek.py
    ```

- **`sft_train/process_sft_data.py`** - 过滤SFT训练数据
  - 从模型响应中提取正确回答并转换为SFT训练格式
  - 需修改代码中的路径：
    - `input_file` - 输入文件路径
    - `output_file` - 输出文件路径
  - 运行示例：
    ```bash
    python sft_train/process_sft_data.py
    ```

### 模型推理与评估

- **`tools/run_qwen_baseline.py`** - Qwen基线模型推理
  - 使用本地Qwen模型对prompts生成回答
  - 需修改代码中的配置：
    - `model_path` - 模型路径
    - `input_file` - 输入prompt文件
    - `output_file` - 输出结果文件
    - `device` - 运行设备
  - 支持断点续传
  - 运行示例：
    ```bash
    python tools/run_qwen_baseline.py
    ```

- **`tools/run_qwen_sft.py`** - SFT微调后的Qwen模型推理
  - 与baseline类似，但使用微调后的模型
  - 需修改代码中的配置：
    - `model_path` - SFT模型路径
    - `input_file` - 输入prompt文件
    - `output_file` - 输出结果文件
    - `device` - 运行设备
  - 运行示例：
    ```bash
    python tools/run_qwen_sft.py
    ```

- **`tools/run_qwen_sft_once.py`** - 单次快速测试脚本
  - 快速测试checkpoint生成效果
  - 主要参数：
    - `--model_dir` - 模型目录
    - `--resume_from` - checkpoint路径或'last'
    - `--device` - 运行设备
    - `--dtype` - 数据类型（bf16/fp16）
  - 运行示例：
    ```bash
    python tools/run_qwen_sft_once.py \
      --model_dir /path/to/model \
      --resume_from /path/to/checkpoint
    ```

- **`tools/run_r1.py`** - DeepSeek R1 API评估
  - 使用DeepSeek R1 API评估模型性能
  - 主要参数：
    - `--dataset_dir` - 数据集目录
    - `--test_samples` - 测试样本数量
    - `--api_key` - API密钥
    - `--api_base` - API地址
    - `--model_name` - 模型名称
    - `--output_dir` - 输出目录
  - 运行示例：
    ```bash
    python tools/run_r1.py \
      --dataset_dir /data2/hongdeyao/Musical_Instruments_0_2022-10-2023-10 \
      --test_samples 200
    ```

- **`compare_model/deepseek.py`** - DeepSeek API评估
  - 使用DeepSeek API评估模型在测试集上的表现，与训练评估逻辑完全一致
  - 主要参数：
    - `--dataset_dir` - 数据集目录
    - `--test_samples` - 测试样本数量
    - `--api_key` - API密钥
    - `--api_base` - API地址
    - `--model_name` - 模型名称
    - `--output_dir` - 输出目录
    - `--test_only` - 仅测试测试集
    - `--candidate_mode` - 候选模式（random/json）
    - `--candidate_json_path` - 候选集JSON路径
  - 支持增量保存
  - 运行示例：
    ```bash
    python compare_model/deepseek.py \
      --test_samples 200 \
      --dataset_dir /data2/hongdeyao/Musical_Instruments_0_2022-10-2023-10 \
      --test_only \
      --candidate_mode json \
      --candidate_json_path /path/to/our_rec/result/json
    ```

- **`compare_model/rrec.py`** - RRec模型训练一致性评估
  - 使用训练时完全一致的评估流程测试checkpoint
  - 主要参数：
    - `--checkpoint_path` - checkpoint路径
    - `--dataset_dir` - 数据集目录
    - `--num_samples` - 评估样本数量
    - `--device` - 运行设备
    - `--reference_json` - 参考JSON文件（可选，用于固定候选集）
    - `--user_window_size` - 用户历史窗口大小
  - 运行示例：
    ```bash
    python compare_model/rrec.py \
      --checkpoint_path /path/to/ckpt \
      --dataset_dir /data2/hongdeyao/Musical_Instruments_0_2022-10-2023-10 \
      --num_samples 100 \
      --reference_json /path/to/our_rec/result/json
    ```

### 模型训练

- **`train.py`** - PPO强化学习训练
  - 基于PL-ranking的PPO训练，使用NDCG@K作为奖励
  - 主要参数：
    - `--model_name_or_path` - 预训练模型路径
    - `--dataset_path` - 数据集路径
    - `--dataset_type` - 数据集类型（movies/other）
    - `--batch_size` - 批次大小
    - `--lr_actor` - Actor学习率
    - `--lr_critic` - Critic学习率
    - `--K` - 候选物品数量
    - `--K_eval` - 评估的top-K
    - `--output_dir` - 输出目录
    - `--resume_from` - 恢复训练的checkpoint
    - `--max_len_per_item` - 每个item最多占用的token
    - `--user_window_size` - 用户历史窗口大小
    - `--print_every` - 打印示例输出的频率
  - 支持分布式训练(DDP)、梯度累积、flash attention
  - 运行示例：
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python train.py \
      --model_name_or_path /path/to/model \
      --dataset_path /data2/hongdeyao/Musical_Instruments_0_2022-10-2023-10 \
      --dataset_type other \
      --output_dir ./outputs
    ```

- **`test.py`** - 模型测试评估
  - 在测试集上评估训练好的模型
  - 主要参数：
    - `--model_name_or_path` - 基础模型路径
    - `--dataset_path` - 数据集路径
    - `--dataset_type` - 数据集类型（movies/other）
    - `--resume_from` - checkpoint路径或'last'
    - `--num_samples` - 测试样本数量
    - `--output_dir` - 输出目录
    - `--mode` - 评估模式（rollout/greedy）
    - `--gen_mode` - 生成展示模式（top1/random/all）
    - `--print_every_text` - 打印示例输出的频率
  - 运行示例：
    ```bash
    python test.py \
      --model_name_or_path /path/to/model \
      --dataset_path /data2/hongdeyao/Musical_Instruments_0_2022-10-2023-10 \
      --resume_from /path/to/checkpoint \
      --num_samples 200
    ```

### 配置文件

- **`paths.py`** - 模型路径配置
  - 定义各模型的本地路径（Qwen2.5-3B-Instruct等）

- **`requirements.txt`** - Python依赖列表
  - 包含所有必需的Python包及版本

- **`train.sh`** / **`test.sh`** - 训练和测试脚本
  - 方便快速启动训练和测试流程

### 目录结构

- **`models/`** - 模型定义
  - `abstract_models.py` - 抽象模型基类
  - `qwen_models.py` - Qwen模型封装

- **`prompters/`** - Prompt生成器
  - `abstract_prompter.py` - Prompter基类
  - `rrec_prompter.py` - RRec专用prompter
  - `prompts.py` - Prompt模板

- **`trainers/`** - 训练器
  - `GRecTrainer.py` - 生成式推荐训练器
  - `RecPOTrainer.py` - 偏好优化训练器
  - `utils.py` - 训练工具函数

- **`tools/`** - 数据分析和辅助工具
  - `analyze_processed_dataset.py` - 数据集统计分析
  - `analyze_accuracy_for_run_qwen_sft.py` - 分析SFT模型准确率
  - `check_common_user.py` - 检查共同用户
  - `find_user_history.py` - 查找用户历史
  - `run_qwen_baseline.py` - Qwen基线模型推理
  - `run_qwen_sft.py` - SFT微调后的Qwen模型推理
  - `run_qwen_sft_once.py` - 单次快速测试脚本
  - `run_r1.py` - DeepSeek R1 API评估
  - `run_r1_once.py` - DeepSeek R1单次测试

- **`compare_model/`** - 对比模型评估
  - `deepseek.py` - DeepSeek API评估
  - `rrec.py` - RRec模型训练一致性评估
  - `rrec.sh` - RRec评估脚本

- **`sft_train/`** - SFT训练相关
  - `generate_prompts.py` - 生成训练prompts
  - `get_response_by_deepseek.py` - 使用DeepSeek API生成responses
  - `process_sft_data.py` - 处理SFT训练数据