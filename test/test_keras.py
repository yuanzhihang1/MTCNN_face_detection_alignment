from keras.models import Model
from keras.layers import Input,Conv2D,Dense,Activation,MaxPooling2D,Flatten
import keras.objectives
import numpy as np
import tensorflow as tf
import keras.backend as K

def cls_loss(y_true,y_pred):
    return K.categorical_crossentropy(y_true,y_pred)
def bbox_loss(y_true,y_pred):
    return K.mean(K.square(y_pred-y_true))*tf.cast(tf.not_equal(tf.reduce_mean(y_true),0),tf.float32)

def det1():
    inputs=Input(shape=(12,12,3))
    conv1=Conv2D(10,3,3,border_mode='valid',activation='relu')(inputs)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)
    conv2=Conv2D(16,3,3,border_mode='valid',activation='relu',name='conv2')(pool1)
    conv3=Conv2D(32,3,3,border_mode='valid',activation='relu',name='conv3')(conv2)
    conv4_1=Conv2D(2,1,1,border_mode='same',name='conv4_1')(conv3)
    conv4_2=Conv2D(4,1,1,border_mode='same',name='conv4_2')(conv3)
    conv4_1=Flatten()(conv4_1)
    bbox=Flatten(name='bbox')(conv4_2)
    prob1=Activation('softmax',name='prob')(conv4_1)
    model=Model(inputs,[prob1,bbox],'det1')
    model.compile('adam',[cls_loss,bbox_loss])
    return model

from keras.utils.np_utils import to_categorical

if __name__=="__main__":
    model=det1()
    x = np.random.random((100, 12, 12, 3))
    y1 = np.random.randint(2, size=(100, 1))
    y2 = np.random.random((100, 4))
    y2=np.zeros((100,4),dtype=np.float32)
    y1 = to_categorical((y1))
    model.fit(x,[y1,y2],nb_epoch=1000)
    model.save('det1')