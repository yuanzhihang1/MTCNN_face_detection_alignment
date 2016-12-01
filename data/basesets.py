import os
import cv2
import numpy as np
import face_pb2 as pb


class Image():
    def __init__(self, pb_obj):
        self.obj = pb_obj
        self.len = len(self.obj)
        self.point = 0

    def reset(self):
        self.point = 0

    def get_one(self,cls):
        assert self.len>0,"There is nothing in the Image repository"
        image = self.obj[self.point]
        self.point += 1
        if self.point>=self.len:
            self.reset()
        assert len(image.dim) == 3, "The image's dimension is not 3"
        data = np.fromstring(image.data, dtype=np.uint8)
        dim = image.dim
        if len(image.shift)!=0:
            shift=np.fromstring(image.shift,dtype=np.int32)
            assert len(shift)==4, "The dimension of image.shift is not 4: "+str(shift)
        else:
            shift=np.zeros([4],dtype=np.int32)
        return [data.reshape(dim),cls,shift]

    def get_batch(self,batch_size,cls):
        batch=[]
        for i in xrange(batch_size):
            batch.append(self.get_one(cls))
        return batch

class Basesets():

    def __init__(self,pt_file_dir):
        self.db=pb.Datasets()
        with open(pt_file_dir,'rb') as f:
            self.db.ParseFromString(f.read())
        self.face = Image(self.db.face)
        self.part_face = Image(self.db.part_face)
        self.back = Image(self.db.back)

    def get_batch(self,face_size,part_face_size,back_size):
        x=[]
        y1=[]
        y2=[]
        for i in self.face.get_batch(face_size,1)+\
               self.part_face.get_batch(part_face_size,1)+\
               self.back.get_batch(back_size,0):
            x.append(i[0])
            y1.append(i[1])
            y2.append(i[2])
        return np.array(x),np.array(y1),np.array(y2)

def mk_pt():
    a=pb.Datasets()
    n=np.arange(200,dtype=np.uint8).reshape([10,10,2])
    im=pb.Image()
    im.data=n.tostring()
    im.dim.extend(n.shape)
    a.face.extend([im])
    a.back.extend([im])
    a.part_face.extend([im])
    with open('test.pt','wb') as f:
        f.write(a.SerializeToString())

if __name__=='__main__':
    mk_pt()
    bs=Basesets('test.pt')
    bt=bs.get_batch(1,0,1)
    pass

