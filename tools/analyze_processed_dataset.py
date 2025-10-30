import os
from itertools import chain

import pandas as pd
from datasets import DatasetDict


def analyze_split(split_name, dataset):
    df = dataset.to_pandas()
    user_count = df["user_id"].nunique()
    sample_count = len(df)

    # 当前交互时间戳
    timestamps = pd.to_datetime(df["timestamp"], unit="ms")

    # 历史交互时间戳，列表需要展开
    history_lists = df["history_timestamp"].dropna().tolist()
    history_flat = list(chain.from_iterable(history_lists)) if history_lists else []
    if history_flat:
        history_timestamps = pd.to_datetime(history_flat, unit="ms")
        min_time = min(history_timestamps.min(), timestamps.min())
        max_time = max(history_timestamps.max(), timestamps.max())
    else:
        min_time = timestamps.min()
        max_time = timestamps.max()

    print(f"[{split_name}] 用户数(唯一): {user_count}")
    print(f"[{split_name}] 样本数(总计): {sample_count}")
    print(f"[{split_name}] 交互历史时间范围: {min_time} ~ {max_time}\n")


def main(dataset_dir):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"找不到数据目录: {dataset_dir}")

    dataset_dict = DatasetDict.load_from_disk(dataset_dir)

    for split_name in ["train", "valid", "test"]:
        if split_name not in dataset_dict:
            print(f"跳过 {split_name}，未在保存的数据集中找到该划分")
            continue
        analyze_split(split_name, dataset_dict[split_name])


if __name__ == "__main__":
    target_dir = "/data/hongdeyao/Musical_Instruments_0_2022-10-2023-10"
    main(target_dir)
