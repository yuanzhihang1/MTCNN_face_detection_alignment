import proto.net_param as pb
import numpy as np
import caffe

caffemodel='mx_mtcnn/model/det1.caffemodel'
prototxt='mx_mtcnn/model/det1.prototxt'


#process the prototxt for new structure
net_spec=pb.Net(prototxt,'prototxt')
#remove the old layers
net_spec.remove_layer('conv4-1')
net_spec.remove_layer('conv4-2')
net_spec.remove_layer('prob1')
conv4=pb.Layer_param('conv4','Convolution',['conv4'],['conv3'])
conv4.conv_param(6,[1],[1])
net_spec.add_layer(conv4,after='PReLU3')
fc1=pb.Layer_param('fc1','InnerProduct',['fc1'],['conv4'])
fc1.fc_param(6)
net_spec.add_layer(fc1,after='conv4')
fc2=pb.Layer_param('fc2','InnerProduct',['fc2'],['fc1'])
fc2.fc_param(6)
net_spec.add_layer(fc2,after='fc1')

net_spec.save_prototxt('../tmp/det1d.prototxt')

#generate the new generated caffemodel with new weights
net=pb.Net(caffemodel)
aw,ab=net.get_layer_data('conv4-1')
bw,bb=net.get_layer_data('conv4-2')
w=np.concatenate((aw,bw),0)
b=np.concatenate((ab,bb),0)
net.add_layer_with_data(conv4,[w,b])
eye=np.eye(6)
zeros=np.zeros([6])
net.add_layer_with_data(fc1,[eye,zeros])
net.add_layer_with_data(fc2,[eye,zeros])
net.save('../tmp/tmp.caffemodel')

net=pb.Caffe('../tmp/det1d.prototxt','../tmp/tmp.caffemodel')
net.net.save('../tmp/det1d.caffemodel')
pass