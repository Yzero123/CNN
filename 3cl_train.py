import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import pathlib

# 图片多分类问题

data_dir = "img_data"
data_dir = pathlib.Path(data_dir) #定义数据集的路径，并将其转换为pathlib.Path对象，便于后续操作。

# 查看数据的长度
image_count = len(list(data_dir.glob('*/*.jpg')))
print("图像总数:", image_count) #使用pathlib遍历data_dir目录下的所有.jpg文件，计算图像总数并打印。

# 开始处理输入的数据集
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
# 使用tf.keras.utils.image_dataset_from_directory函数从data_dir目录中加载图像数据集，并将其分为训练集和验证集。validation_split参数指定了验证集的比例，subset参数指定了是训练集还是验证集。image_size参数指定了图像的大小，batch_size指定了每个批次的图像数量。

# 获取数据集中所有类别的名称，并打印出来。
class_names = train_ds.class_names
num_classes = len(class_names)
print("类别名称:", class_names) 

# 开始建模
model = models.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
# 构建一个顺序模型（Sequential），并添加一系列层。这些层包括：
# Rescaling层：将图像像素值缩放到0到1之间。
# Conv2D层：卷积层，用于提取图像特征。
# MaxPooling2D层：池化层，用于降低特征图的维度。
# Dropout层：防止过拟合。
# Flatten层：将多维特征图展平为一维。
# Dense层：全连接层，用于分类。

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 使用adam优化器、sparse_categorical_crossentropy损失函数和accuracy指标编译模型。sparse_categorical_crossentropy用于多分类问题，其中标签是整数。

# 训练模型
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
# 使用model.fit方法训练模型，传入训练数据集和验证数据集，以及训练的周期数（epochs）

# 保存模型
model.save("model/3cl_epoch10.h5")  # 保存模型到文件