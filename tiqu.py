import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from data_set import *
from modelres import *
from load_image import *
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
from keras import backend as K



os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True  
sess = tf.Session(config=config)
KTF.set_session(sess)


model = resnet()
#装载训练好的模型,这里是使用keras_retinanet工程编写的load_model函数，实际上也是调用了keras.models import load_model方法
model.load_weights('pretrained_weights.hdf5')
 
#读取图片
images=cv2.imread("0.jpg",cv2.IMREAD_GRAYSCALE)
#cv2.imshow("Image", images)
print(images.shape)

image = np.array(images)
h, w= image.shape[0],image.shape[1]
image = Image.fromarray(image)
images = image.resize((int(h/2),int(w/2)),Image.ANTIALIAS)

 
#扩展图像的维度
image_arr = np.expand_dims(images, axis=0)
image_arr = np.reshape(image_arr,image_arr.shape + (1,))
print(image_arr.shape)

# 第一个参数,表示模型的输入层，[model.layers[0].input]表示输入数据；
# 第二个参数,表示模型的输出层，可根据需要提取模型相应的中间层作为输出层，如[model.layers[2].output]表示获取中间层第2层作为输出层
# 注意，layer_n实际上是一个函数
layer_n = K.function([model.layers[0].input], [model.layers[1].output])
 
#通过输入端输入数据，获取模型的输出结果
f1 = layer_n([image_arr])[0]
print(f1)
# 根据模型输出层的特征数，遍历输出层的所有特征图像(通常输出层是多通道的，不能直接显示出来)
row_col = math.floor(f1.shape[3] ** 0.5)
for _ in range(row_col*row_col):
    show_img = f1[:, :, :, _]
    #print(f1.shape)
    feature_shape=show_img.shape
    #再次调整特征图像的维度，调整为2维特征
    show_img.shape = [feature_shape[1], feature_shape[2]]
    #根据特征数目计算显示的格子个数
    plt.subplot(row_col, row_col, _ + 1)
    #将图像投影到plt画布上
    plt.imshow(show_img)
    #关闭坐标
    plt.axis('off')
plt.show()

    
