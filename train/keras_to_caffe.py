<<<<<<< HEAD
import det1
import numpy as np
import nn_tools.keras_to_caffe as convertor

if __name__=='__main__':
    det1=det1.det1()
    det1.load_weights('det1.kmodel')
    net=convertor.convert_kmodel(det1)
    net.
    pass
=======
import keras
import proto.net_param as pb
import det1

net=pb.Net()
keras_model=det1.model()
>>>>>>> origin/master
