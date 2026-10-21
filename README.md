# 🐶 CNN Image Classification

> A complete, runnable image classification project: a small CNN built with Keras, trained on a 3-class dataset, with the trained weights committed so you can predict **immediately after cloning** — no training required.

[![CI](https://github.com/Yzero123/CNN/actions/workflows/ci.yml/badge.svg)](https://github.com/Yzero123/CNN/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://www.tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/test%20accuracy-99.53%25-brightgreen.svg)](#-measured-results)

[中文](README.zh-CN.md) · [Measured results](#-measured-results) · [Quick start](#-quick-start) · [CLI reference](#-cli-reference) · [Dataset](#-dataset) · [Project layout](#-project-layout) · [Limitations](#-limitations) · [Roadmap](#-roadmap)

---

## 📊 Measured results

The numbers below are **measured, not estimated**. Reproduce them yourself:

```bash
pip install -r requirements.txt
python evaluate.py
```

Against the committed test set of **214 images**:

| Metric | Value |
| --- | --- |
| Overall accuracy | **99.53%** (213 / 214) |
| car | 87/87 — **100.00%** |
| chicken | 57/58 — **98.28%** |
| dog | 69/69 — **100.00%** |

**Confusion matrix** (rows = truth, columns = prediction):

|  | car | chicken | dog |
| --- | ---: | ---: | ---: |
| **car** | 87 | 0 | 0 |
| **chicken** | 0 | 57 | 1 |
| **dog** | 0 | 0 | 69 |

The single miss is one chicken image predicted as dog.

### Why this matters: the previous weights were broken

The repository used to ship `model/3cl_epoch10.h5`, trained for 10 epochs. Measured on the same test set, it scored **67.29%** overall — and the breakdown showed the real story:

| Class | Old model (10 epochs) | New model (20 epochs) |
| --- | ---: | ---: |
| car | 100.00% | 100.00% |
| chicken | 98.28% | 98.28% |
| **dog** | **0.00%** (0 / 69) | **100.00%** (69 / 69) |
| Overall | 67.29% | **99.53%** |

The old model **never once predicted "dog"** — every one of the 69 dog images was labelled car (17) or chicken (52). The overall 67.29% was a flattering average over two easy classes and one completely dead class.

Root cause: the model was simply **under-trained**. The labels were correct, and the architecture was fine — 10 epochs was not enough for the dog class to converge, while the visually distinctive car and chicken classes had already saturated. Training the same code for 20 epochs on the same data took dog from 0/69 to 69/69. The old weights have been removed from the working tree; they remain in the git history if you want to compare.

---

## 🚀 Quick start

### Predict with the committed model (no training needed)

```bash
git clone https://github.com/Yzero123/CNN.git
cd CNN

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python predict.py                          # a bundled test image
python predict.py path/to/your/image.jpg   # your own image
```

Output:

```
使用类别：['car', 'chicken', 'dog']
car的概率为0.00%
chicken的概率为0.00%
dog的概率为100.00%
最终预测的结果为：dog
```

### Train on the bundled dataset

```bash
python 3cl_train.py                          # 20 epochs, same settings that produced the committed model
python 3cl_train.py --epochs 30 --batch-size 16
```

Training prints a summary and saves the weights:

```
===== 训练结果 =====
训练损失: 最后一轮 = 0.0000  最佳 = 0.0000
训练准确率: 最后一轮 = 1.0000  最佳 = 1.0000
验证损失: 最后一轮 = 0.1528  最佳 = 0.0992
验证准确率: 最后一轮 = 0.9965  最佳 = 0.9965

模型已保存到：model/3cl_epoch20.h5
```

Random sources are seeded (`--seed`, default 123), so the same command reproduces the committed weights and the numbers above.

### Use your own dataset

Put your images in one folder per class and point the scripts at it. **The category names are the class labels, ordered alphabetically** — the scripts derive them from the directory structure, so no code change is needed:

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

## 📖 CLI reference

### `3cl_train.py`

| Option | Default | Description |
| --- | --- | --- |
| `--data-dir` | `img_data` | Dataset root; one sub-folder per class |
| `--epochs` | `20` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--image-size` | `180` | Input image edge length in pixels |
| `--validation-split` | `0.2` | Fraction held out for validation |
| `--seed` | `123` | Random seed for the train/validation split |
| `--model-out` | `model/3cl_epoch20.h5` | Where to save the trained model |

### `predict.py`

| Argument | Default | Description |
| --- | --- | --- |
| `image` (positional) | `img_test/dog/1001.jpg` | Image to classify |
| `--model` | `model/3cl_epoch20.h5` | Model file to load |
| `--data-dir` | `img_data` | Used to derive the class-name ordering |

### `evaluate.py`

| Option | Default | Description |
| --- | --- | --- |
| `--model` | `model/3cl_epoch20.h5` | Model to evaluate |
| `--test-dir` | `img_test` | Test set root |
| `--data-dir` | `img_data` | Used to derive the class-name ordering |
| `--image-size` | `180` | Input image edge length in pixels |
| `--json` | off | Print machine-readable JSON |

---

## 🗂️ Dataset

| Split | car | chicken | dog | Total |
| --- | ---: | ---: | ---: | ---: |
| `img_data/` (training) | 953 | 923 | 976 | 2,852 |
| `img_test/` (evaluation) | 87 | 58 | 69 | 214 |

Images are 1920×1080 for the car and dog classes and mixed-resolution for the chicken class; every image is resized to 180×180 during loading.

Four images in the test set are truncated JPEGs. `evaluate.py` detects them, falls back to PIL's truncated-tolerant mode, and lists them in its output rather than crashing.

---

## 🧠 Model architecture

```
1. Rescaling(1./255)   - normalise pixels to 0-1
2. Conv2D(16)          - extract low-level features
3. MaxPooling2D()
4. Conv2D(32)          - extract mid-level features
5. MaxPooling2D()
6. Conv2D(64)          - extract high-level features
7. MaxPooling2D()
8. Dropout(0.2)        - regularisation
9. Flatten
10. Dense(128)         - fully connected
11. Dense(num_classes) - softmax output
```

Compiled with `adam`, `sparse_categorical_crossentropy`, and accuracy as the metric.

---

## 📂 Project layout

```
CNN/
├── img_data/            # training images, one folder per class
├── img_test/            # test images, one folder per class
├── model/
│   └── 3cl_epoch20.h5   # trained weights (46 MB), committed for instant use
├── 3cl_train.py         # training script
├── predict.py           # single-image prediction
├── evaluate.py          # accuracy / confusion matrix on the test set
├── requirements.txt     # pinned dependency ranges
├── .github/workflows/ci.yml
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE              # MIT
├── README.md
└── README.zh-CN.md
```

---

## ⚠️ Limitations

Read this before you quote the 99.53% figure:

- **The test set is in-distribution.** `img_test/` was drawn from the same source pool as `img_data/` (the same numbering scheme, 1001–1070 versus 1–999), so the number measures fit on that pool — not generalisation to entirely new imagery. Expect a noticeably lower score on photos from a different source.
- **The training set has low diversity.** Every car and dog image is exactly 1920×1080, which suggests many frames come from a small number of sources. A few near-duplicate and at least one non-photographic sample (a papier-mâché dog figurine) are present in `img_data/dog/`.
- **Training accuracy reaches 100%** within a few epochs. The model memorises the training set, so the number of epochs past convergence buys little.
- **Loading an `.h5` file executes code.** Keras 2.x HDF5 weights are not safe to load from an untrusted source. See [SECURITY.md](SECURITY.md).

---

## 🗺️ Roadmap

- [x] Commit the trained weights so the project is usable immediately
- [x] Argument parsing for training and prediction, so a custom dataset needs no code edits
- [x] `evaluate.py` for reproducible accuracy and confusion-matrix reporting
- [x] Retrain with enough epochs for every class to converge (dog: 0% → 100%)
- [x] Handle truncated JPEGs gracefully instead of crashing
- [ ] Data augmentation (flip / rotation / zoom) to fight the overfitting
- [ ] Replace the HDF5 weights with the safer `.keras` format
- [ ] Expand to more classes and a larger, more diverse dataset
- [ ] A small unit-test suite for the scripts themselves

---

## 🤝 Contributing

Issues and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). If you change the model or the training recipe, please include the measured accuracy from `python evaluate.py`.

---

## 📄 License

MIT License — see the [LICENSE](LICENSE) file.

Dataset images come from public material sources and are included for learning and demonstration. The trademarks, portraits and other rights in those images remain with their respective owners; the repository does not grant redistribution rights for them.
