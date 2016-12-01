from keras.models import Model
from keras.layers import Input,Conv2D,Dense,Activation,MaxPooling2D,Flatten
import keras.objectives
import numpy as np
import tensorflow as tf
import keras.backend as K
import data.wider_data as wider_data
from keras.utils.np_utils import to_categorical

def cls_loss(y_true,y_pred):
    pi=tf.log(y_pred)
    return tf.reduce_sum(-tf.cast(tf.not_equal(y_true,-1),tf.float32)*(y_true*pi-(1-y_true)*(1-pi)))

def bbox_loss(y_true,y_pred):
<<<<<<< HEAD
    # return keras.objectives.mean_squared_error(y_true,y_pred)/5
    return keras.objectives.mean_squared_error(y_true,y_pred) * tf.cast(tf.not_equal(tf.reduce_mean(y_true),0),tf.float32)/5
=======
    return K.mean(K.square(y_pred-y_true))*tf.cast(tf.not_equal(tf.reduce_mean(y_true),0),tf.float32)
>>>>>>> origin/master

def det1():
    inputs=Input(shape=(12,12,3),name='input')
    conv1=Conv2D(10,3,3,border_mode='valid',activation='relu',name='conv1')(inputs)
    pool1=MaxPooling2D(pool_size=(2,2),name='pool1')(conv1)
    conv2=Conv2D(16,3,3,border_mode='valid',activation='relu',name='conv2')(pool1)
    conv3=Conv2D(32,3,3,border_mode='valid',activation='relu',name='conv3')(conv2)
    conv4_1=Conv2D(2,1,1,border_mode='same',name='conv4_1')(conv3)
    conv4_2=Conv2D(4,1,1,border_mode='same',name='conv4_2')(conv3)
    conv4_1=Flatten()(conv4_1)
    bbox=Flatten(name='bbox')(conv4_2)
    prob1=Activation('softmax',name='prob')(conv4_1)
    model=Model(inputs,[prob1,bbox],'det1')
<<<<<<< HEAD
    model.compile('adam',[keras.objectives.binary_crossentropy,bbox_loss])
    return model

if __name__=="__main__":
    with tf.device('/gpu:0'):
        data=wider_data.Wider('../widerL/JPEGImages','../widerL/Annotations')
=======
    model.compile('adam',[keras.objectives.binary_crossentropy,keras.objectives.mean_squared_error])
    return model

if __name__=="__main__":
    with tf.device('/cpu:0'):
        data=wider_data.Wider('widerL/JPEGImages','widerL/Annotations')
>>>>>>> origin/master
        model=det1()
        #model.load_weights('det1_2.kmodel')
        s=model.get_weights()
        #y1 = to_categorical((y1))

        # for i in range(300):
        #     x, y1, y2 = data.get_batch(100, 100, 100)
        #     loss=model.train_on_batch(x,[y1,y2])
        #     print loss
        for i in range(10):
            x, y1, y2 = data.get_batch(10000, 10000, 20000)
            x = (x - 127.5) * 0.0078125
            y1=to_categorical(y1)
            model.fit(x,[y1,y2],nb_epoch=3,batch_size=32)
<<<<<<< HEAD
        model.save('det1.kmodel')
=======
        model.save('det1_2.kmodel')
>>>>>>> origin/master
