import keras.backend as k 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import utils as np_utils
import numpy as np 
from load_image import load_dataset, segmentation

np.random.seed(1)

IMAGE_SIZE = 256

class Dataset:
    def __init__(self, path_name,label_name):
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.path_name = path_name
        self.label_name = label_name
        self.input_shape = None

    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=1, nb_classes=2):
        images, labels = load_dataset(self.path_name,self.label_name)
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)
        if k.image_data_format() == 'channels_first':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)   
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)
        #train_labels = np_utils.to_categorical(train_labels, nb_classes)
        #test_labels = np_utils.to_categorical(test_labels, nb_classes)

        train_labels = np.reshape(train_labels,train_labels.shape + (1,))
        test_labels = np.reshape(test_labels,test_labels.shape + (1,))

        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')

        train_images /= 255
        test_images /= 255

        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels

        return train_images,train_labels,test_images,test_labels

