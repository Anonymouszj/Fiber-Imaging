import matplotlib.pyplot as plt
import numpy as np
from data_set import *
from model import *
from load_image import *
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
from sklearn.metrics import precision_recall_curve
from sklearn.utils.multiclass import type_of_target

np.random.seed(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True  
sess = tf.Session(config=config)
KTF.set_session(sess)


train_data = Dataset('/home/server/zj/dense/test/','/home/server/zj/dense/testlabel/')

train_images, train_labels,test_images, test_labels = Dataset.load(train_data)

# model is defined in model.py
model = get_model_deep_speckle()
# pretrained_weights.hdf5 can be downloaded from the link on our GitHub project page
model.load_weights('pretrained_weights.hdf5')
'''
speckle = np.load('/home/server/zj/liqing/test3.npy')
pred_speckle_E = model.predict(speckle, batch_size=2)

plt.figure()
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(speckle[i, :].squeeze(), cmap='hot')
    plt.axis('off')
    plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(pred_speckle_E[i, :, :, 0].squeeze(), cmap='gray')
    plt.axis('off')

plt.show()

''' 

results = model.predict(train_images,verbose=1)

#saveResult("/home/server/zj/dense/text",results)

results = results.flatten()
train_labels = train_labels.flatten()

plt.figure()
plt.title('P-R')
plt.xlabel('Recall')
plt.ylabel('Precision')
precision,recall, thresholds = precision_recall_curve(train_labels,results)
plt.plot(precision,recall)
plt.show()
plt.savefig('PR.png')