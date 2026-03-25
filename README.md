# CNN Image Classification

A simple image multi-classification project based on CNN and TensorFlow framework.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Project Introduction

This is an image multi-classification learning project using Convolutional Neural Network (CNN) and TensorFlow framework.

### What You Can Learn

- How to build CNN models with TensorFlow
- How to process image datasets
- How to train and predict with models
- CNN basics: Convolution, Pooling, Dense layers

---

## Project Structure

```
CNN/
├── img_data/          # Training data directory (organized by class)
├── img_test/          # Test data directory
├── model/             # Model save directory
├── 3cl_train.py       # Model training script
├── predict.py         # Prediction script
└── README.md          # Project documentation
```

---

## Tech Stack

| Technology | Description |
|------------|-------------|
| Python | Programming Language |
| TensorFlow / Keras | Deep Learning Framework |
| NumPy | Numerical Computing |
| OpenCV | Image Processing |
| Matplotlib | Data Visualization |

---

## Model Architecture

This project uses a classic CNN structure:

```
1. Rescaling Layer    - Pixel normalization (0-1)
2. Conv2D(16)         - Convolution layer, extract basic features
3. MaxPooling2D        - Pooling, reduce dimensions
4. Conv2D(32)         - Convolution layer, extract mid-level features
5. MaxPooling2D        - Pooling
6. Conv2D(64)         - Convolution layer, extract high-level features
7. MaxPooling2D        - Pooling
8. Dropout(0.2)       - Prevent overfitting
9. Flatten            - Flatten feature maps
10. Dense(128)        - Fully connected layer
11. Dense(num_classes) - Output layer (Softmax)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| batch_size | 32 | Number of training samples per batch |
| img_height | 180 | Image height |
| img_width | 180 | Image width |
| epochs | 10 | Number of training epochs |
| validation_split | 0.2 | Validation split ratio |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow numpy opencv-python matplotlib
```

### 2. Prepare Data

Put your images into `img_data/` directory, organized by class:

```
img_data/
├── class_1/
│   ├── image1.jpg
│   └── ...
├── class_2/
│   └── ...
└── class_3/
    └── ...
```

### 3. Train Model

```bash
python 3cl_train.py
```

### 4. Make Predictions

```bash
python predict.py
```

---

## Notes

1. **Image Format**: Supports .jpg, .jpeg, .png formats
2. **Image Size**: Automatically resized to (180, 180)
3. **Data Amount**: Recommend at least 100 images per class for better results

---

## Contact

- **QQ**: 248119587

---

## License

MIT License
