import numpy as np
import tensorflow as tf
import cv2
import sys
import os
import xml.etree.ElementTree as ET
import time

anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

def logistic_activate(x):
    return 1.0 / (1.0 + np.exp(-x))

class EVA_TF:
    clean_up = False
    test_file = '2007_test.txt'
    disp_console = True
    box_iou_threshold = 0.45
    prob_threshold = .005
    input_size_ = 416
    coord_ = 4
    side_ = 13
    num = 5
    classes_ = 20
    BBNum = 169
    mAP = []

    classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

    def __init__(self,sess,image, train_phase,  target,argvs = []):
        self.argv_parser(argvs)
        self.sess = sess
        self.image = image
        self.target = target
        self.train_phase = train_phase
        self.test_demo()
    def argv_parser(self,argvs):
        for i in range(1,len(argvs),2):
            if argvs[i] == '-disp_console' :
                if argvs[i+1] == '1' :self.disp_console = True
                else : self.disp_console = False
                

    def convertdetections(self, predictions):
        probs = np.zeros(shape=(self.side_, self.side_, self.num, self.classes_), dtype=np.float32)
        boxes = np.zeros(shape=(self.side_, self.side_,self.num, 4), dtype=np.float32)
        for i in range(0,self.BBNum):
            row = i / self.side_
            col = i % self.side_
            for n in range(0,self.num):
                scale = predictions[row, col, n*25+4]
                boxes[row, col, n, 0] = (col + logistic_activate(predictions[row, col, n*25])) / self.side_
                boxes[row, col, n, 1] = (row + logistic_activate(predictions[row, col, n*25+1])) / self.side_
                boxes[row, col, n, 2] = (np.exp(predictions[row, col, n*25+2])) * anchors[2*n] / self.side_
                boxes[row, col, n, 3] = (np.exp(predictions[row, col, n*25+3])) * anchors[2*n+1] / self.side_
                for j in range(0,self.classes_):
                    prob =  scale * predictions[row, col, n*25+5+j]
                    if prob > self.prob_threshold:
                        probs[row, col, n, j] = prob
        return probs.reshape((self.BBNum*self.num, self.classes_)), boxes.reshape((self.BBNum*self.num, 4))

    def overlap( self, x1,  w1,  x2,  w2):
        l1 = x1 - w1/2
        l2 = x2 - w2/2
        left = max(l1,l2)
        r1 = x1 + w1/2
        r2 = x2 + w2/2
        right = min(r1,r2)
        return right - left

    def box_iou(self,a,b):
        w = self.overlap(a[0], a[2], b[0], b[2])
        h = self.overlap(a[1], a[3], b[1], b[3])
        if(w <=  0 or h <= 0):
            return 0
        inter = w*h
        union = a[2]*a[3] + b[2]*b[3] - inter
        return inter/union

    def do_nms_sort(self, boxes, probs):
        s = []
        total =self.BBNum*self.num
        for i in range(0,total):
            s.append((i,probs[i]))

        for k in range(0,self.classes_):
            s = sorted(s,reverse=True,key=lambda x:x[1][k])
            for i in range(0,total):
                if probs[s[i][0]][k]==0:
                    continue
                boxA = boxes[s[i][0]]
                for j in range(i+1,total):
                    boxB = boxes[s[j][0]]
                    if probs[s[j][0]][k]==0 :
                        continue
                    if self.box_iou(boxA,boxB)>self.box_iou_threshold:
                        probs[s[j][0]][k] = 0
        return probs

    def write_voc_results_file(self, all_boxes):
        if self.disp_console: print "write voc result file..."
        cwd = os.getcwd()
        if not os.path.isdir(cwd+'/results'):
            os.mkdir(cwd+'/results')
        for cls_ind, cls in enumerate(self.classes):
            path = os.path.join(cwd,'results', 'via_det_val_{:s}.txt')
            filename = path.format(cls)
            with open(filename, 'w') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue

                    for k in xrange(dets.shape[0]):
                        index = index.split("/")[-1].split(".")[0]
                        xmin = dets[k, 0] - dets[k, 2]/2.
                        xmax = dets[k, 0] + dets[k, 2]/2.
                        ymin = dets[k, 1] - dets[k, 3]/2.
                        ymax = dets[k, 1] + dets[k, 3]/2.

                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                            xmin, ymin, xmax, ymax))

    def parse_rec(self, imagepath):
        tmppath = imagepath.replace('JPEGImages','Annotations') 
        labelpath = tmppath.replace('jpg', 'xml')

        tree = ET.parse(labelpath)
        objects= []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            obj_struct['bbox'] = [xmin,ymin,xmax,ymax]
            obj_struct['difficult'] = int(obj.find('difficult').text)
            objects.append(obj_struct)
        return objects

    def do_eval(self):
        cwd = os.getcwd()
        for i, cls in enumerate(self.classes):
            path = os.path.join(cwd,'results', 'via_det_val_{:s}.txt')
            filename = path.format(cls)


            #read ground truth
            class_recs = {}
            npos = 0

            recs = {}
            for j , imagename in enumerate(self.image_index):
                recs[imagename] = self.parse_rec(imagename)
                R = [obj for obj in recs[imagename] if obj['name'] == cls]
                bbox = np.array([x['bbox'] for x in R])
                difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
                det = [False] * len(R)
                #npos = npos + len(R)
                npos = npos +sum(~difficult)
                index = self.image_index[j].split("/")[-1].split(".")[0]
                class_recs[index]= {'bbox': bbox,
                                    'difficult': difficult,
                                    'det': det}


            #read dets
            with open(filename) as f:
                lines = f.readlines()

            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            #sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)

            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)

                if BBGT.size > 0:
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + 
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > 0.5:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.

                else:
                    fp[d] = 1.


            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)

            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.voc_ap(rec, prec)
	    precision = tp[-1]/float(fp[-1] + tp[-1])
	    recall = tp[-1]/ float(npos)
            self.mAP += [ap]
            print "class ", cls ,"ap= ", ap, "precision= ", precision, "recall= ", recall


    def voc_ap(self, rec, prec):
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        return ap

    def do_clean_up(self):
        if self.disp_console: print "do_clean_up..."    
        for cls in self.classes:
            filename = "results/via_det_val_{:s}.txt".format(cls)
            os.remove(filename)




    def test_demo(self):
        if self.disp_console : print "test_demo..."
        with open(self.test_file) as f:
            self.image_index = [x.strip('\n') for x in f]

        num_images = len(self.image_index)

        all_boxes = [[[] for _ in xrange(num_images)]
                         for _ in xrange(len(self.classes))]
        for i in range(num_images):
            print "i = ", i
            img = cv2.imread(self.image_index[i])
            self.h_img,self.w_img,_ = img.shape
            resize_size = (self.input_size_, self.input_size_)
            img_resized = cv2.resize(img, resize_size)
            crop_img = img_resized/255.0
            image_in = np.asarray( crop_img )
            dataset = np.ndarray(shape=(1, self.input_size_, self.input_size_, 3), dtype=np.float32)
            dataset[0] = np.float32(image_in)

            feed_dict = {self.image : dataset, self.train_phase : False}
            net_output = self.sess.run(self.target, feed_dict=feed_dict)
            probs,boxes = self.convertdetections(net_output[0])
            probs = self.do_nms_sort(boxes, probs)

            result = []
            for j in range(0,self.BBNum*self.num):
                cls =np.argmax(probs[j])
                prob = probs[j][cls]
                if prob < self.prob_threshold:
                    continue
                result.append([cls, boxes[j][0]*self.w_img, boxes[j][1]*self.h_img, boxes[j][2]*self.w_img, boxes[j][3]*self.h_img, prob])
            

            if len(result) == 0:
                continue
            result = np.array(result)
            for k in xrange(len(self.classes)):
                inds = np.where(result[:,0] == k)[0]
                cls_dets = result[inds,1:]
                all_boxes[k][i] = cls_dets

            
        self.write_voc_results_file(all_boxes)
        self.do_eval()
        print "mAP= ", np.mean(self.mAP)
        if self.clean_up:
            self.do_clean_up()
'''
    def test_demo_2(self):
        if self.disp_console : print "test_demo..."
        with open(self.test_file) as f:
            self.image_index = [x.strip('\n') for x in f]

        num_images = len(self.image_index)

        all_boxes = [[[] for _ in xrange(num_images)]
                         for _ in xrange(len(self.classes))]
        for i in range(num_images):
            print "i = ", i
            image_data = cv2.imread(self.image_index[i])
            self.h_img,self.w_img,_ = image_data.shape
	    crop_img =2*image_data/255.0-1
	    if image_data.shape[0]<image_data.shape[1]:
		pad_top = (image_data.shape[1]-image_data.shape[0])/2
		pad_botton = pad_top
		pad_left=0
		pad_right = 0
	    else:
		pad_left = (image_data.shape[0]-image_data.shape[1])/2
		pad_right = pad_left
		pad_top=0
		pad_botton = 0
	    crop_img = cv2.copyMakeBorder(crop_img, pad_top,pad_botton,pad_left,pad_right, cv2.BORDER_CONSTANT, value=0)
	    ResizeSize = (self.input_size_, self.input_size_)
	    crop_img = cv2.resize(crop_img,ResizeSize)

            image_in = np.asarray( crop_img )
            dataset = np.ndarray(shape=(1, self.input_size_, self.input_size_, 3), dtype=np.float32)
            dataset[0] = np.float32(image_in)

            feed_dict = {self.image : dataset}
            net_output = self.sess.run(self.target, feed_dict=feed_dict)
            probs,boxes = self.convertdetections(net_output[0])
            probs = self.do_nms_sort(boxes, probs)

            result = []
            for j in range(0,self.BBNum*self.num):
                cls =np.argmax(probs[j])
                prob = probs[j][cls]
                if prob < 0.3:
                    continue
		s = max((image_data.shape[0],image_data.shape[1]))
		boxes[j][0] = (boxes[j][0]*s-pad_left)/(s-2*pad_left)
		boxes[j][1] = (boxes[j][1]*s-pad_top)/(s-2*pad_top)
		boxes[j][2] = boxes[j][2]*s/(s-2*pad_left)
		boxes[j][3] = boxes[j][3]*s/(s-2*pad_top)
                result.append([cls, boxes[j][0]*self.w_img, boxes[j][1]*self.h_img, boxes[j][2]*self.w_img, boxes[j][3]*self.h_img, prob])
            

            if len(result) == 0:
                continue
            result = np.array(result)
            for k in xrange(len(self.classes)):
                inds = np.where(result[:,0] == k)[0]
                cls_dets = result[inds,1:]
                all_boxes[k][i] = cls_dets

            
        self.write_voc_results_file(all_boxes)
        self.do_eval()
        if self.clean_up:
            self.do_clean_up()
'''
