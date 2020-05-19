import numpy as np
import os
import sys
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import skimage.io as io

np.random.seed(1)

def down_sampling(image):
    image = np.array(image)
    h, w= image.shape[0],image.shape[1]
    image = Image.fromarray(image)
    image = image.resize((int(h/2),int(w/2)),Image.ANTIALIAS)
    return image

def read_path(path_name,label_name):
    label_index = 0
    images = []
    labels = []
    path = []
    path1 = []
    for dir_item in os.listdir(path_name):
            path.append(dir_item)
            full_path = os.path.abspath(os.path.join(path_name,dir_item))
            if os.path.isdir(full_path):
                read_path(full_path,None)
            else:
                if dir_item.endswith('.jpg'):
                    image = cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)
                    #image = cv2.flip(image, 1)
                    image = down_sampling(image)
                    image = np.array(image)
                else:
                    break
                images.append(image)

    for dir_item1 in os.listdir(label_name):
            path1.append(dir_item1)
            full_path1 = os.path.abspath(os.path.join(label_name,dir_item1))
            if os.path.isdir(full_path1):
                read_path(None,full_path1)
            else:
                if dir_item.endswith('.jpg'):
                    label = cv2.imread(full_path1,cv2.IMREAD_GRAYSCALE)
                    label = segmentation(label)
                    label = down_sampling(label)
                    label = np.array(label)
                else:
                    break  
                labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images,labels

def segmentation(image):
    _, image = cv2.threshold(image,0,1,cv2.THRESH_BINARY)
    return image

def load_dataset(path_name,label_name):   
    images, labels = read_path(path_name,label_name)
    images = np.array(images)
    return images,labels

def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile):
        img=item[:,:,0]
        #img[img>=0.5]=1
        #img[img<0.5]=0
        io.imsave(os.path.join(save_path,"%d_predict.jpg"%i),img)

if __name__ == "__main__":
    path_name = '/home/server/zj/user2/mnist/'
    label_name ='/home/server/zj/user2/label/'
    load_dataset(path_name,label_name)