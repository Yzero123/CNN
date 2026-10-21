# 🐶 CNN 图片多分类

> 一个开箱即用的图片分类项目：用 Keras 搭一个小 CNN，在 3 类数据集上训练，**训练好的权重直接随仓库分发**——克隆下来就能预测，不用先训练。

[![CI](https://github.com/Yzero123/CNN/actions/workflows/ci.yml/badge.svg)](https://github.com/Yzero123/CNN/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://www.tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/test%20accuracy-99.53%25-brightgreen.svg)](#-实测结果)

[English](README.md) · [实测结果](#-实测结果) · [快速开始](#-快速开始) · [命令参考](#-命令参考) · [数据集](#-数据集) · [项目结构](#-项目结构) · [已知局限](#-已知局限) · [路线图](#-路线图)

---

## 📊 实测结果

下面的数字是**实测的，不是估的**。你可以自己复现：

```bash
pip install -r requirements.txt
python evaluate.py
```

在随仓库提供的 **214 张**测试图上：

| 指标 | 数值 |
| --- | --- |
| 总体准确率 | **99.53%**（213 / 214） |
| car | 87/87 —— **100.00%** |
| chicken | 57/58 —— **98.28%** |
| dog | 69/69 —— **100.00%** |

**混淆矩阵**（行 = 真实，列 = 预测）：

|  | car | chicken | dog |
| --- | ---: | ---: | ---: |
| **car** | 87 | 0 | 0 |
| **chicken** | 0 | 57 | 1 |
| **dog** | 0 | 0 | 69 |

唯一的错判是一张鸡被认成了狗。

### 为什么这件事值得写出来：旧权重是坏的

仓库此前提供的是 `model/3cl_epoch10.h5`，只训练了 10 轮。在同一测试集上实测，**总体准确率 67.29%**——但分项才是真相：

| 类别 | 旧模型（10 轮） | 新模型（20 轮） |
| --- | ---: | ---: |
| car | 100.00% | 100.00% |
| chicken | 98.28% | 98.28% |
| **dog** | **0.00%**（0 / 69） | **100.00%**（69 / 69） |
| 总体 | 67.29% | **99.53%** |

旧模型**从来没有一次预测出 "dog"**——69 张狗全被判成 car（17 张）或 chicken（52 张）。67.29% 这个总体数字是被两个好类别和一个完全死掉的类别平均出来的虚高值。

根因是**训练不足**，不是标签错也不是代码错：同样这套代码、同一份数据，训练到 20 轮后 dog 从 0/69 变成 69/69。旧权重已从工作区移除，但仍保留在 git 历史里，随时可以对比。

---

## 🚀 快速开始

### 用仓库自带模型预测（无需训练）

```bash
git clone https://github.com/Yzero123/CNN.git
cd CNN

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python predict.py                          # 用仓库里的测试图
python predict.py path/to/your/image.jpg   # 用你自己的图
```

输出：

```
使用类别：['car', 'chicken', 'dog']
car的概率为0.00%
chicken的概率为0.00%
dog的概率为100.00%
最终预测的结果为：dog
```

### 用自带数据集训练

```bash
python 3cl_train.py                          # 20 轮，与生成仓库权重时的设置一致
python 3cl_train.py --epochs 30 --batch-size 16
```

训练结束会打印摘要并保存权重：

```
===== 训练结果 =====
训练损失: 最后一轮 = 0.0000  最佳 = 0.0000
训练准确率: 最后一轮 = 1.0000  最佳 = 1.0000
验证损失: 最后一轮 = 0.1528  最佳 = 0.0992
验证准确率: 最后一轮 = 0.9965  最佳 = 0.9965

模型已保存到：model/3cl_epoch20.h5
```

随机源已固定（`--seed`，默认 123），同样的命令可以复现出仓库里的权重和上面这组数字。

### 换成自己的数据集

按类别建子目录即可。**目录名就是类别标签，按字母序排列**——脚本会自动从目录结构推导标签，不需要改任何代码：

```
my_dataset/
├── cat/
│   ├── 1.jpg
│   └── 2.jpg
├── bird/
│   └── 1.jpg
└── fish/
    └── 1.jpg
```

```bash
python 3cl_train.py --data-dir my_dataset --model-out model/my_model.h5
python predict.py path/to/image.jpg --data-dir my_dataset --model model/my_model.h5
```

---

## 📖 命令参考

### `3cl_train.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data-dir` | `img_data` | 数据集根目录，每个子目录一个类别 |
| `--epochs` | `20` | 训练轮数 |
| `--batch-size` | `32` | 批大小 |
| `--image-size` | `180` | 输入图片边长（像素） |
| `--validation-split` | `0.2` | 验证集比例 |
| `--seed` | `123` | 训练/验证划分的随机种子 |
| `--model-out` | `model/3cl_epoch20.h5` | 模型保存路径 |

### `predict.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `image`（位置参数） | `img_test/dog/1001.jpg` | 待预测图片 |
| `--model` | `model/3cl_epoch20.h5` | 加载的模型文件 |
| `--data-dir` | `img_data` | 用于推导类别顺序 |

### `evaluate.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `model/3cl_epoch20.h5` | 待评估模型 |
| `--test-dir` | `img_test` | 测试集根目录 |
| `--data-dir` | `img_data` | 用于推导类别顺序 |
| `--image-size` | `180` | 输入图片边长（像素） |
| `--json` | 关闭 | 输出机器可读 JSON |

---

## 🗂️ 数据集

| 划分 | car | chicken | dog | 合计 |
| --- | ---: | ---: | ---: | ---: |
| `img_data/`（训练） | 953 | 923 | 976 | 2,852 |
| `img_test/`（评估） | 87 | 58 | 69 | 214 |

car 与 dog 两类的图片全部是 1920×1080，chicken 一类的尺寸则不统一；加载时统一缩放到 180×180。

测试集里有 4 张 JPEG 是截断的。`evaluate.py` 会识别它们，退回到 PIL 的容忍截断模式读取，并在输出里列出来，而不是直接崩掉。

---

## 🧠 模型结构

```
1. Rescaling(1./255)   - 像素归一化到 0-1
2. Conv2D(16)          - 提取低层特征
3. MaxPooling2D()
4. Conv2D(32)          - 提取中层特征
5. MaxPooling2D()
6. Conv2D(64)          - 提取高层特征
7. MaxPooling2D()
8. Dropout(0.2)        - 正则化
9. Flatten
10. Dense(128)         - 全连接
11. Dense(num_classes) - softmax 输出
```

优化器 `adam`，损失函数 `sparse_categorical_crossentropy`，评估指标为准确率。

---

## 📂 项目结构

```
CNN/
├── img_data/            # 训练集，按类别分子目录
├── img_test/            # 测试集，按类别分子目录
├── model/
│   └── 3cl_epoch20.h5   # 训练好的权重（46 MB），随仓库分发，克隆即可用
├── 3cl_train.py         # 训练脚本
├── predict.py           # 单图预测
├── evaluate.py          # 测试集上的准确率与混淆矩阵
├── requirements.txt     # 依赖版本范围
├── .github/workflows/ci.yml
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE              # MIT
├── README.md
└── README.zh-CN.md
```

---

## ⚠️ 已知局限

引用 99.53% 这个数字之前请先读这段：

- **测试集与训练集同分布。** `img_test/` 与 `img_data/` 来自同一批素材（编号规则相同，1001–1070 对 1–999），所以这个数字衡量的是「对这一批素材的拟合程度」，不是对全新图片的泛化能力。换一批来源的照片，准确率会明显下降。
- **训练集多样性偏低。** car 与 dog 的图片尺寸全部是 1920×1080，说明大量样本来自少数几个来源；`img_data/dog/` 里还存在近似重复的样本，以及至少一张非实拍样本（一个纸艺狗摆件）。
- **训练准确率几轮就达到 100%。** 模型把训练集背下来了，所以收敛之后再加轮数收益很小。
- **加载 `.h5` 文件等同于执行代码。** Keras 2.x 的 HDF5 权重不能从不可信来源加载，详见 [SECURITY.md](SECURITY.md)。

---

## 🗺️ 路线图

- [x] 随仓库分发训练好的权重，克隆即可用
- [x] 训练与预测脚本支持命令行参数，换数据集不用改代码
- [x] `evaluate.py` 提供可复现的准确率与混淆矩阵报告
- [x] 重新训练到各类别都收敛（dog：0% → 100%）
- [x] 优雅处理截断的 JPEG 而不是崩溃
- [ ] 数据增强（翻转 / 旋转 / 缩放），缓解过拟合
- [ ] 把 HDF5 权重换成更安全的 `.keras` 格式
- [ ] 扩充类别与数据集规模、多样性
- [ ] 为脚本本身补一套单元测试

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request，详见 [CONTRIBUTING.md](CONTRIBUTING.md)。如果你改了模型或训练配方，请在 PR 里附上 `python evaluate.py` 的实测结果。

---

## 📄 License

MIT License —— 详见 [LICENSE](LICENSE) 文件。

数据集图片来自公开素材源，仅用于学习与演示。这些图片中的商标、肖像等权利仍归各自权利人所有，本仓库不授予对其的再分发权利。
