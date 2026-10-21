#!/usr/bin/env python3
"""
评估脚本 - 在测试集上计算准确率、每类召回率与混淆矩阵

用法示例：
    python evaluate.py                      # 评估默认模型与默认测试集
    python evaluate.py --model model/3cl_epoch20.h5
    python evaluate.py --test-dir img_test
    python evaluate.py --json               # 输出机器可读结果

README 中引用的准确率就是用这个脚本算出来的，可以直接复现。
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile

DEFAULT_DATA_DIR = "img_data"
DEFAULT_TEST_DIR = "img_test"
DEFAULT_MODEL_PATH = "model/3cl_epoch20.h5"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_image(path, size):
    """加载图片。

    数据集里有几张 JPEG 被截断，严格模式下 PIL 会直接抛错。
    这里先按正常方式加载，失败时退回到「容忍截断」模式，
    并把这类文件记录下来提醒使用者。
    """
    try:
        return tf.keras.utils.load_img(path, target_size=size), False
    except OSError:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            return tf.keras.utils.load_img(path, target_size=size), True
        except Exception:
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            raise


def parse_args():
    parser = argparse.ArgumentParser(description="评估 CNN 模型在测试集上的表现")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help=f"模型路径（默认 {DEFAULT_MODEL_PATH}）")
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR, help=f"测试集目录（默认 {DEFAULT_TEST_DIR}）")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help=f"训练集目录，用于推导类别顺序（默认 {DEFAULT_DATA_DIR}）")
    parser.add_argument("--image-size", type=int, default=180, help="输入图片边长（默认 180）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    return parser.parse_args()


def derive_class_names(data_dir):
    """从数据集目录的子目录名推导类别列表（按字母序，与训练时一致）。"""
    if not os.path.isdir(data_dir):
        return None
    names = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    return names or None


def collect_samples(test_dir, class_names):
    """收集 (真实标签, 图片路径) 列表。"""
    samples = []
    for cls in class_names:
        folder = os.path.join(test_dir, cls)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            samples.append((cls, os.path.join(folder, name)))
    return samples


def main():
    args = parse_args()

    class_names = derive_class_names(args.data_dir)
    if not class_names:
        sys.exit(f"无法从 {args.data_dir} 推导类别名，请检查目录结构。")

    if not os.path.isfile(args.model):
        sys.exit(f"找不到模型文件：{args.model}")

    model = tf.keras.models.load_model(args.model)
    samples = collect_samples(args.test_dir, class_names)
    if not samples:
        sys.exit(f"{args.test_dir} 下没有找到测试图片。")

    per_class = {c: {"correct": 0, "total": 0} for c in class_names}
    confusion = collections.Counter()
    predictions = collections.Counter()
    truncated = []

    for truth, path in samples:
        img, was_truncated = load_image(path, (args.image_size, args.image_size))
        if was_truncated:
            truncated.append(path)
        arr = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)
        pred = class_names[int(np.argmax(model.predict(arr, verbose=0)[0]))]

        per_class[truth]["total"] += 1
        predictions[pred] += 1
        confusion[(truth, pred)] += 1
        if pred == truth:
            per_class[truth]["correct"] += 1

    total = len(samples)
    correct = sum(v["correct"] for v in per_class.values())
    overall = correct / total * 100

    result = {
        "model": args.model,
        "test_dir": args.test_dir,
        "samples": total,
        "correct": correct,
        "overall_accuracy": round(overall, 2),
        "per_class": {
            c: {
                "correct": per_class[c]["correct"],
                "total": per_class[c]["total"],
                "accuracy": round(per_class[c]["correct"] / per_class[c]["total"] * 100, 2),
            }
            for c in class_names
        },
        "confusion_matrix": {
            c: {p: confusion.get((c, p), 0) for p in class_names}
            for c in class_names
        },
        "prediction_distribution": dict(predictions),
        "truncated_images": truncated,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"模型：{args.model}")
    print(f"测试集：{args.test_dir}")
    print(f"类别顺序：{class_names}")
    print()
    print(f"总体准确率：{correct}/{total} = {overall:.2f}%")
    print()
    print("每类准确率：")
    for c in class_names:
        info = per_class[c]
        acc = info["correct"] / info["total"] * 100 if info["total"] else 0.0
        print(f"  {c:<10} {info['correct']:>4}/{info['total']:<4} = {acc:>6.2f}%")
    print()
    print("预测分布：" + str(dict(predictions)))
    print()
    print("混淆矩阵（行 = 真实，列 = 预测）：")
    print(" " * 12 + "".join(f"{p:>10}" for p in class_names))
    for c in class_names:
        print(f"  {c:<10}" + "".join(f"{confusion.get((c, p), 0):>10}" for p in class_names))
    if truncated:
        print()
        print(f"⚠️  {len(truncated)} 张测试图文件被截断（已按容忍截断模式读取）：")
        for p in truncated:
            print(f"   - {p}")


if __name__ == "__main__":
    main()
