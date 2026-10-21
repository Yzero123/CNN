#!/usr/bin/env python3
"""
训练脚本 - 图片多分类（CNN + TensorFlow）

用法示例：
    python 3cl_train.py
    python 3cl_train.py --epochs 20 --batch-size 16
    python 3cl_train.py --data-dir path/to/dataset --model-out model/my_model.h5
"""

import argparse
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

DEFAULT_DATA_DIR = "img_data"
DEFAULT_MODEL_PATH = "model/3cl_epoch20.h5"


def set_seed(seed):
    """固定随机源，让同样的参数能训练出同一个模型。"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="训练图片多分类 CNN 模型")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help=f"数据集目录（默认 {DEFAULT_DATA_DIR}）")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数（默认 10）")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小（默认 32）")
    parser.add_argument("--image-size", type=int, default=180, help="输入图片边长（默认 180）")
    parser.add_argument("--validation-split", type=float, default=0.2, help="验证集比例（默认 0.2）")
    parser.add_argument("--seed", type=int, default=123, help="随机种子（默认 123）")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH, help=f"模型保存路径（默认 {DEFAULT_MODEL_PATH}）")
    return parser.parse_args()


def build_model(num_classes):
    """构建顺序模型：Rescaling → 3×(Conv2D+MaxPooling) → Dropout → Flatten → Dense。"""
    model = models.Sequential([
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main():
    args = parse_args()

    set_seed(args.seed)

    # 定义数据集的路径，并转换为 pathlib.Path 对象，便于后续操作
    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(
            f"找不到数据集目录：{data_dir}\n"
            f"请把图片按类别放进 {data_dir}/<类别名>/ 目录下，或用 --data-dir 指定。"
        )

    # 查看数据的长度
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print("图像总数:", image_count)
    if image_count == 0:
        raise SystemExit(f"{data_dir} 下没有找到 .jpg 图片，请检查数据集结构。")

    # 开始处理输入的数据集
    image_size = (args.image_size, args.image_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=args.validation_split,
        subset="training",
        seed=args.seed,
        image_size=image_size,
        batch_size=args.batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=args.validation_split,
        subset="validation",
        seed=args.seed,
        image_size=image_size,
        batch_size=args.batch_size)

    # 获取数据集中所有类别的名称，并打印出来
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("类别名称:", class_names)

    # 开始建模
    model = build_model(num_classes)
    model.summary()

    # 训练模型
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )

    # 输出训练结果摘要
    print("\n===== 训练结果 =====")
    labels = {
        "loss": "训练损失",
        "accuracy": "训练准确率",
        "val_loss": "验证损失",
        "val_accuracy": "验证准确率",
    }
    for metric in ("loss", "accuracy", "val_loss", "val_accuracy"):
        values = history.history.get(metric)
        if not values:
            continue
        # 损失越低越好，其余指标越高越好
        best = min(values) if metric.endswith("loss") else max(values)
        print(f"{labels[metric]}: 最后一轮 = {values[-1]:.4f}  最佳 = {best:.4f}")

    # 保存模型
    model_path = pathlib.Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"\n模型已保存到：{model_path}")
    print(f"使用 `python predict.py --model {model_path}` 进行预测。")


if __name__ == "__main__":
    main()
