import tensorflow as tf
import numpy as np
import cv2

# 加载模型
model_path = "model/3cl_epoch10.h5"  # 确保这里的路径与保存时的路径一致
model = tf.keras.models.load_model(model_path)

# 预测图片
def predict_image(image_path, model, class_names):
    # 用tf的工具来处理输入图片
    img = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  

    # 预测的结果为一个概率表[[],[]]
    result_p = model.predict(img_array)[0]
    
    # 打印每个类别的概率
    for p, class_name in zip(result_p, class_names):
        print("{}的概率为{:.2f}%".format(class_name, p * 100))
    
    # 取出概率最大的值所在索引，并映射出原来的分类标准
    max_index = np.argmax(result_p)
    print("最终预测的结果为：{}".format(class_names[max_index]))

# 调用预测函数
dog_01_path = "img_test/dog/1001.jpg"
class_names = ['car', 'chicken', 'dog']  # 根据你的类别来设置
predict_image(dog_01_path, model, class_names)