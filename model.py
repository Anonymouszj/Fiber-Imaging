from __future__ import print_function

from keras.callbacks import LearningRateScheduler, ModelCheckpoint,TensorBoard
from keras.layers import (Activation, Concatenate, Conv2D, Dropout, Input,
                          MaxPooling2D, UpSampling2D)
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import *
from keras.regularizers import l2

np.random.seed(1)

def conv_factory(x, concat_axis, nb_filter,
                 dropout_rate=None, weight_decay=1E-4):
    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (5, 5), dilation_rate=(2, 2),
               kernel_initializer="he_uniform",
               padding="same",
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x

def denseblock(x, concat_axis, nb_layers, growth_rate,
               dropout_rate=None, weight_decay=1E-4):
    list_feat = [x]
    for i in range(nb_layers):
        x = conv_factory(x, concat_axis, growth_rate,
                         dropout_rate, weight_decay)
        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)

    return x

def get_model_deep_speckle(pretrained_weights = None,input_size=(256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(inputs)
    db1 = denseblock(x=conv1, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5)
    pool1 = MaxPooling2D(pool_size=(2, 2))(db1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same',kernel_initializer='he_normal')(pool1)
    db2 = denseblock(x=conv2, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5)
    pool2 = MaxPooling2D(pool_size=(2, 2))(db2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(pool2)
    db3 = denseblock(x=conv3, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5)
    pool3 = MaxPooling2D(pool_size=(2, 2))(db3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    db4 = denseblock(x=conv4, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5)
    pool4 = MaxPooling2D(pool_size=(2, 2))(db4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(pool4)
    db5 = denseblock(x=conv5, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5)
    up5 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(db5))
    merge5 = Concatenate(axis=3)([db4, up5])

    conv6 = Conv2D(512, 3, activation='relu', padding='same',kernel_initializer='he_normal')(merge5)
    db6 = denseblock(x=conv6, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5)
    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(db6))
    merge6 = Concatenate(axis=3)([db3, up6])

    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    db7 = denseblock(x=conv7, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5)
    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(db7))
    merge7 = Concatenate(axis=3)([db2, up7])

    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    db8 = denseblock(x=conv8, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5)
    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(db8))
    merge8 = Concatenate(axis=3)([db1, up8])

    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    db9 = denseblock(x=conv9, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5)
    conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(db9)
    conv11 = Conv2D(1, 1, activation='sigmoid')(conv10)

    model = Model(inputs=inputs, outputs=conv11)

    model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
    #categorical_crossentropy
    #model.summary()
    
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
        
    return model

if __name__ == "__main__":
    model = get_model_deep_speckle()
