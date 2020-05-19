from model import *
import numpy as np
import keras.backend as k 
from data_set import Dataset
import matplotlib.pyplot as plt
from load_image import saveResult   
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os

np.random.seed(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True  
sess = tf.Session(config=config)
KTF.set_session(sess)

train_data = Dataset('/home/server/zj/dense/train/','/home/server/zj/dense/trainlabel/')

train_images, train_labels,test_images, test_labels = Dataset.load(train_data)

model = get_model_deep_speckle()
#model = get_model_deep_speckle('pretrained_weights.hdf5')

tensorboard=TensorBoard(log_dir='/home/server/zj/dense/log',write_images=True,histogram_freq=0,write_grads=True)

checkpoint = ModelCheckpoint('pretrained_weights.hdf5', monitor='loss',verbose=1, save_best_only=True)

callback_lists=[tensorboard,checkpoint]

model.fit(train_images,train_labels,batch_size=1,epochs=100,validation_data=(test_images,test_labels),callbacks=callback_lists,verbose=1,shuffle=True)
#,validation_data=(test_images,test_labels)
model.save_weights("model.h5")
