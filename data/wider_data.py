import os
import xml.dom.minidom as dom
import cv2
import numpy as np
import basesets
import random
import face_pb2 as pb
import sys

class Wider(basesets.Basesets):
    MIN_DET_SIZE=12.
    thresh_positive = 0.65
    thresh_negtive = 0.3
    thresh_part_face=0.4

    def __init__(self, image_dir, annotation_dir):
        self.image_dir = [os.path.join(image_dir, dir) for dir in os.listdir(image_dir)]
        self.annotation_dir = [os.path.join(annotation_dir, name.split('.')[0] + '.xml') for name in
                               os.listdir(image_dir)]
        self.point = 0
        self.num_images = len(self.image_dir)
        self.name = 'wider'
        cache_path=os.path.join(os.path.split(__file__)[0],'Cache','%s.pt'%self.name)
        if not os.path.exists(cache_path):
            self.make_cache(cache_path)
        basesets.Basesets.__init__(self,cache_path)


    def get_xml_annotation(self, xml_dir):
        file = dom.parse(xml_dir)
        root = file.documentElement
        boxs = []
        for obj in root.getElementsByTagName('object'):
            temp_box = []
            box = obj.getElementsByTagName('bndbox')[0]
            for i in ['xmin', 'ymin', 'xmax', 'ymax']:
                temp_box.append(int(box.getElementsByTagName(i)[0].childNodes[0].nodeValue))
            boxs.append(temp_box)
        return boxs

    def read_one_raw_data(self):
        im=cv2.imread(self.image_dir[self.point])
        gt_boxs=self.get_xml_annotation(self.annotation_dir[self.point])
        self.point+=1
        if self.point>=self.num_images:
            self.point=0
        face, part_face, back = self.generate_from_im(im, gt_boxs)
        return face, part_face, back

    def make_cache(self,file_name):
        def extend(pool,waters):
            for im, iou,shift in waters:
                db_im = pb.Image()
                db_im.data = im.tostring()
                db_im.iou = iou
                db_im.dim.extend(im.shape)
                if shift!=[]:
                    db_im.shift=shift.tostring()
                pool.extend([db_im])
        db=pb.Datasets()
        self.point=350
        for i in range(10000):
            print "making cache at",i
            face, part_face, back =self.read_one_raw_data()
            extend(db.face,face)
            extend(db.part_face,part_face)
            extend(db.back,back)
        with open(file_name,'wb') as f:
            f.write(db.SerializeToString())

    def show(self,im,boxs):
        for box in boxs:
            cv2.rectangle(im, tuple(box[0:2]),tuple(box[2:]),(0,255,0  ))
        cv2.imshow('image',im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def generate_from_im(self, im, gt_boxs, minsize=20, factor=0.7):
        """
        :param min_det_size:
        :param minsize: minimal face to detect
        :param factor:
        :return:
        """
        # min_det_size is the det size
        min_det_size = self.MIN_DET_SIZE
        minl = min(*im.shape[:2])
        scales = []
        m = min_det_size / minsize
        minl = m * minl
        factor_count = 0
        while minl > min_det_size:
            scales.append(m * factor ** factor_count)
            minl *= factor
            factor_count += 1
        #self.show(im,gt_boxs)
        face = []
        gt_boxs=np.array(gt_boxs)
        for gt in gt_boxs:
            if gt[0]<gt[2] and gt[1]<gt[3]:
                r=np.sum(gt[2:]-gt[:2])/4
                if r<=8:
                    np.delete(gt_boxs,gt)
                    continue
                ctr=(gt[2:]+gt[:2])/2
                xy=np.concatenate((ctr-r,ctr+r),0)
                xy=np.maximum(0,xy)
                xy=np.minimum([im.shape[1],im.shape[0],im.shape[1],im.shape[0]],xy).astype(np.int)
                if np.sum(xy[2:]-xy[:2]<=0)!=2:
                    continue
                #cv2.imwrite('../tmp/x.jpg', cv2.resize(im[xy[1]:xy[3],xy[0]:xy[2]], (48, 48)))
                face.append([cv2.resize(im[xy[1]:xy[3],xy[0]:xy[2]],(int(self.MIN_DET_SIZE),int(self.MIN_DET_SIZE))),1.,[]])
        part_face = []
        back = []
        np.random.shuffle(scales)
        for scale in scales:
            resized_im = cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)))
            i=0
            while(len(face)<len(gt_boxs)*3 and len(part_face)<len(gt_boxs)*3):
                x=random.randint(self.MIN_DET_SIZE,resized_im.shape[1])
                y=random.randint(self.MIN_DET_SIZE,resized_im.shape[0])
                pbox = np.array([x - min_det_size, y - min_det_size, x, y], dtype=float) / scale
                iou=0
                for gt_box in gt_boxs:
                    new_iou=self.iou(pbox,gt_box)
                    if new_iou>iou:
                        iou=new_iou
                        shift=np.array((gt_box-pbox)*scale,dtype=np.int32)
                        pass
                i+=1
                if i>300:
                    break
                if iou==0:
                    continue
                croped_im = resized_im[y - min_det_size:y,x - min_det_size:x]
                assert croped_im.shape==(min_det_size,min_det_size,3)
                if 0<iou<self.thresh_negtive and len(back)<len(gt_boxs)*10:
                    back.append([croped_im,iou,[]])
                if iou>self.thresh_positive:
                    face.append([croped_im,iou,[]])
                if self.thresh_positive>iou>self.thresh_part_face:
                    part_face.append([croped_im,iou,shift])
        return face,part_face,back

    def get_label_and_shift(self, pbox, gt_boxs, thresh_positive=0.65, thresh_negtive=0.3):
        """
        :param pbox: [x0,x1,y0,y1]
        :param gt_boxs: with format [(x0,x1,y0,y1)]
        :return:
        """
        pbox = np.array(pbox)
        gt_boxs = np.array(gt_boxs)
        iou = 0
        shift = [0, 0, 0, 0]
        for gt_box in gt_boxs:
            iiou = self.iou(pbox, gt_box)
            if iou < iiou:
                iou = iiou
                shift = gt_box - pbox
        if iou > thresh_positive:
            return 1, shift
        if iou < thresh_negtive:
            return 0, [0, 0, 0, 0]
        else:
            return -1, shift

    def iou(self, box1, box2):
        """
        :param box1: foramt:[x0,y0,x1,y1]
        :param box2: foramt:[x0,y0,x1,y1]
        :return: the iou based on box2
        """
        box1 = np.array(box1)
        box2 = np.array(box2)
        box3 = np.vstack((box1, box2))
        S = box3[:, 2:] - box3[:, :2]
        S = S[:, 0] * S[:, 1]

        xy0 = np.maximum(box1[:2], box2[:2])
        xy1 = np.minimum(box1[2:], box2[2:])
        S0 = np.maximum(0, xy1 - xy0)
        S0 = S0[0] * S0[1]
        if S0 != 0:
            pass
        return np.min(float(S0) / S)

class Wider_old():
    MIN_DET_SIZE=12.
    def __init__(self,image_dir,annotation_dir):
        self.image_dir=[os.path.join(image_dir,dir) for dir in os.listdir(image_dir)]
        self.annotation_dir=[os.path.join(annotation_dir,name.split('.')[0]+'.xml') for name in os.listdir(image_dir)]
        self.point=0
        self.num_images=len(self.image_dir)
        self.name='wider'

    def batch(self,batch_size=1):
        ims=[]
        labels=[]
        shifts=[]
        for i in range(batch_size):
            im=cv2.imread(self.image_dir[self.point])
            gt_boxes=self.get_xml_annotation(self.annotation_dir[self.point])
            im,label,shift=self.generate_from_im(im,gt_boxes)
            ims+=im
            labels+=label
            shifts+=shift
            self.point+=1
            if self.point>=self.num_images:
                self.point=0
        return np.array(ims),np.array(labels),np.array(shifts)

    def get_xml_annotation(self,xml_dir):
        file=dom.parse(xml_dir)
        root = file.documentElement
        boxs=[]
        for obj in root.getElementsByTagName('object'):
            temp_box=[]
            box=obj.getElementsByTagName('bndbox')[0]
            for i in ['xmin','ymin','xmax','ymax']:
                temp_box.append(int(box.getElementsByTagName(i)[0].childNodes[0].nodeValue))
            boxs.append(temp_box)
        return boxs

    def generate_from_im(self,im,gt_boxs,minsize=20,factor=0.7):
        """
        :param min_det_size:
        :param minsize: minimal face to detect
        :param factor:
        :return:
        """
        # min_det_size is the det size
        min_det_size = self.MIN_DET_SIZE
        minl = min(*im.shape[:2])
        scales = []
        m = min_det_size / minsize
        minl = m*minl
        factor_count = 0
        while minl > min_det_size:
            scales.append(m * factor ** factor_count)
            minl *= factor
            factor_count += 1
        ims=[]
        labels=[]
        shifts=[]
        for scale in scales:
            resized_im=cv2.resize(im,(int(im.shape[0]*scale),int(im.shape[1]*scale)))
            x=y=0
            while x+min_det_size<resized_im.shape[0]:
                x+=min_det_size
                while y+min_det_size<resized_im.shape[1]:
                    y+=min_det_size
                    croped_im=resized_im[x-min_det_size:x,y-min_det_size:y]
                    ims.append(croped_im)
                    pbox=np.array([x-min_det_size,y-min_det_size,x,y],dtype=float)/scale
                    label,shift=self.get_label_and_shift(pbox,gt_boxs)
                    labels.append(label)
                    shifts.append(shift)
        return ims,labels,shifts

    def get_label_and_shift(self,pbox,gt_boxs,thresh_positive=0.65,thresh_negtive=0.3):
        """
        :param pbox: [x0,x1,y0,y1]
        :param gt_boxs: with format [(x0,x1,y0,y1)]
        :return:
        """
        pbox=np.array(pbox)
        gt_boxs=np.array(gt_boxs)
        iou=0
        shift=[0,0,0,0]
        for gt_box in gt_boxs:
            iiou=self.iou(pbox, gt_box)
            if iou<iiou:
                iou=iiou
                shift = gt_box - pbox
        if iou>thresh_positive:
            return 1,shift
        if iou<thresh_negtive:
            return 0,[0,0,0,0]
        else:
            return -1,shift


    def iou(self,box1,box2):
        """
        :param box1: foramt:[x0,y0,x1,y1]
        :param box2: foramt:[x0,y0,x1,y1]
        :return: the iou based on box2
        """
        box1=np.array(box1)
        box2=np.array(box2)
        box3=np.vstack((box1,box2))
        S=box3[:,2:]-box3[:,:2]
        S=S[:,0]*S[:,1]
        S = np.where(S==0,np.inf,S)

        xy0=np.maximum(box1[:2],box2[:2])
        xy1=np.minimum(box1[2:],box2[2:])
        S0=np.maximum(0,xy1-xy0)
        S0=S0[0]*S0[1]

        return np.min(float(S0)/S)



if __name__=='__main__':
    print __file__
    data=Wider('../widerL/JPEGImages','../widerL/Annotations')
    x,y1,y2=data.get_batch(3,3,3)
    for i,im in enumerate(x):
        cv2.imwrite('../tmp/%d.jpg'%i,im)

