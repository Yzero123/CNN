# CNN Image Classification

基于 CNN 和 TensorFlow 框架的图片多分类项目

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![MIT License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📖 项目简介

这是一个基于卷积神经网络（CNN）和 TensorFlow 框架的图片多分类学习项目。

### 你可以学习到
- 如何使用 TensorFlow 构建 CNN 模型
- 如何处理图像数据集
- 如何进行模型训练和预测
- CNN 的基本结构：卷积层、池化层、全连接层

---

## 🏗️ 项目结构

`
CNN/
├── img_data/          # 训练数据目录（按类别分类）
├── img_test/          # 测试数据目录
├── model/             # 模型保存目录
├── 3cl_train.py       # 模型训练脚本
├── predict.py         # 图片预测脚本
└── README.md          # 项目说明文档
`

---

## 🔧 技术栈

| 技术 | 说明 |
|------|------|
| Python | 编程语言 |
| TensorFlow / Keras | 深度学习框架 |
| NumPy | 数值计算 |
| OpenCV | 图像处理 |
| Matplotlib | 数据可视化 |

---

## 📝 模型架构

本项目使用经典的 CNN 结构：

`
1. Rescaling层     - 像素值归一化 (0-1)
2. Conv2D(16)      - 卷积层，提取基础特征
3. MaxPooling2D    - 池化，降低维度
4. Conv2D(32)      - 卷积层，提取中级特征
5. MaxPooling2D    - 池化
6. Conv2D(64)      - 卷积层，提取高级特征
7. MaxPooling2D    - 池化
8. Dropout(0.2)    - 防止过拟合
9. Flatten         - 展平特征图
10. Dense(128)     - 全连接层
11. Dense(num_classes) - 输出层 (Softmax)
`

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 32 | 每批训练样本数 |
| img_height | 180 | 图片高度 |
| img_width | 180 | 图片宽度 |
| epochs | 10 | 训练轮数 |
| validation_split | 0.2 | 验证集比例 |

---

## 🚀 快速开始

### 1. 安装依赖

\\\ash
pip install tensorflow numpy opencv-python matplotlib
\\\

### 2. 准备数据

将你的图片按类别放入 img_data/ 目录

### 3. 训练模型

\\\ash
python 3cl_train.py
\\\

### 4. 进行预测

\\\ash
python predict.py
\\\

---

## ⚠️ 注意事项

1. **图片格式**：支持 .jpg, .jpeg, .png 等格式
2. **图片尺寸**：会自动 resize 到 (180, 180)
3. **数据量**：每个类别建议至少 100 张图片

---

## 📞 联系方式

- **QQ**: 248119587

---

## 📄 License

MIT License