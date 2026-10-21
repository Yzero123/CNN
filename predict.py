#!/usr/bin/env python3
"""
预测脚本 - 用训练好的模型对图片做分类

用法示例：
    python predict.py                          # 用默认测试图预测
    python predict.py path/to/image.jpg        # 预测指定图片
    python predict.py --data-dir img_data      # 用自定义数据集推导类别名
    python predict.py --model model/3cl_epoch10.h5

类别名默认从 --data-dir 下的子目录名按字母序推导，
因此换成自己的数据集后无需修改代码。
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

DEFAULT_DATA_DIR = "img_data"
DEFAULT_MODEL_PATH = "model/3cl_epoch20.h5"
DEFAULT_TEST_IMAGE = "img_test/dog/1001.jpg"
FALLBACK_CLASS_NAMES = ['car', 'chicken', 'dog']


def derive_class_names(data_dir):
    """从数据集目录的子目录名推导类别列表（按字母序）。"""
    if not os.path.isdir(data_dir):
        return None
    names = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    return names or None


def load_model(model_path):
    """加载模型。"""
    if not os.path.isfile(model_path):
        sys.exit(
            f"找不到模型文件：{model_path}\n"
            f"请先运行 `python 3cl_train.py` 生成模型，或用 --model 指定路径。"
        )
    return tf.keras.models.load_model(model_path)


def predict_image(image_path, model, class_names):
    """预测单张图片并打印各类别概率。"""
    if not os.path.isfile(image_path):
        sys.exit(f"找不到图片文件：{image_path}")

    img = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # 预测结果为一个概率表 [[...]]
    result_p = model.predict(img_array)[0]

    for p, class_name in zip(result_p, class_names):
        print(f"{class_name}的概率为{p * 100:.2f}%")

    max_index = int(np.argmax(result_p))
    print(f"最终预测的结果为：{class_names[max_index]}")
    return class_names[max_index]


def parse_args():
    parser = argparse.ArgumentParser(
        description="用训练好的 CNN 模型预测图片类别",
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_TEST_IMAGE,
        help=f"待预测的图片路径（默认 {DEFAULT_TEST_IMAGE}）",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"模型文件路径（默认 {DEFAULT_MODEL_PATH}）",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"数据集目录，用于推导类别名（默认 {DEFAULT_DATA_DIR}）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    class_names = derive_class_names(args.data_dir) or FALLBACK_CLASS_NAMES
    print(f"使用类别：{class_names}")

    model = load_model(args.model)
    predict_image(args.image, model, class_names)


if __name__ == "__main__":
    main()
