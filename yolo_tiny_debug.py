import cv2
import numpy as np
import tensorflow as tf
import time
import sys
import random
import math
from eval_interface import EVA_TF
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.training import moving_averages
from tensorflow.core.protobuf import saver_pb2
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from numpy import set_printoptions
np.set_printoptions(threshold='nan')

# Used to keep the update ops done by batch_norm.
UPDATE_OPS_COLLECTION = '_update_ops_'
# Collection containing all the variables created using variables
MODEL_VARIABLES = '_model_variables_'
# Collection containing the slim.variables that are created with restore=True.
VARIABLES_TO_RESTORE = '_variables_to_restore_'

coord_ = 4
side_ = 13
num = 5
BBNum = side_*side_
classes_ = 20
deep_ = classes_ + coord_
input_size_ = 416
batch_size = 16
thresh = 0.24
iouThresh = 0.4
train_file1 = "train.txt"
test_file = "2007_test.txt"
TOWER_NAME = 'tower'
anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
B = 5
C = 20
W = 13
H = 13
HW = 13*13
sprob = 1
sconf = 5
snoob = 1
scoor  = 1

def expit_tensor(x):
    return 1. / (1. + tf.exp(-x))

def logistic_activate(x):
    return 1.0 / (1.0 + np.exp(-x))

def distort_image(bgr_image, hue, sat, val):
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # hue
    hsv_image[:, :, 0] += int(hue * 179)

    # sat
    hsv_image[:, :, 1] *= sat

    # val
    hsv_image[:, :, 2] *= val

    hsv_image[hsv_image[:,:,0] > 179, 0] -= 179
    hsv_image[hsv_image[:,:,0] < 0, 0] += 179
    hsv_image[hsv_image > 255] = 255
    hsv_image[hsv_image < 0] = 0
    hsv_image = hsv_image.astype(np.uint8)
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return bgr_image
    

def rand_scale(s):
    scale = np.random.uniform(1, s)
    rand = np.random.randint(0, 2)
    if rand :
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = np.random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    im = distort_image(im, dhue, dsat, dexp)
    return im

def imcv2_recolor(im, a = .1):
    t = [np.random.uniform()]
    t += [np.random.uniform()]
    t += [np.random.uniform()]
    t = np.array(t) * 2. - 1.

    # random amplify each channel
    im = im * (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform() * 2 - 1
    im = np.power(im/mx, 1. + up * .5)
    return np.array(im * 255., np.uint8)

def random_hsv_image(bgr_image, delta_hue, delta_sat_scale, delta_val_scale):
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # hue
    hsv_image[:, :, 0] += int((np.random.rand() * delta_hue * 2 - delta_hue) * 255)

    # sat
    sat_scale = 1 + np.random.rand() * delta_sat_scale * 2 - delta_sat_scale
    hsv_image[:, :, 1] *= sat_scale

    # val
    val_scale = 1 + np.random.rand() * delta_val_scale * 2 - delta_val_scale
    hsv_image[:, :, 2] *= val_scale

    hsv_image[hsv_image < 0] = 0 
    hsv_image[hsv_image > 255] = 255 
    hsv_image = hsv_image.astype(np.uint8)
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return bgr_image


def generate_posarray():
    base = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    a = [base] * 13
    b = np.expand_dims(base, 1)
    b = np.concatenate([b] * 13, 1)
    a = np.expand_dims(a, 2)
    b = np.expand_dims(b, 2)
    return np.concatenate([a, b], 2)

def coord_iou(coords, _coords):
    wh = coords[:, :, :, 2:4]
    area = wh[:, :, :, 0] * wh[:, :, :, 1]
    centers = coords[:, :, :, 0:2]
    transform_array = tf.reshape(tf.expand_dims(tf.constant(generate_posarray(), dtype=tf.float32), dim=2), [HW,1,2])
    centers = (centers + transform_array) / 13.
    floor = centers - (wh * .5)
    ceil = centers + (wh * .5)

    _wh = _coords[:, :, :, 2:4]
    _area = _wh[:, :, :, 0] * _wh[:, :, :, 1]
    _centers = _coords[:, : , :, 0:2]
    _centers= (_centers + transform_array) / 13.
    _floor = _centers - (_wh * .5)
    _ceil = _centers + (_wh * .5)

    intersect_upleft   = tf.maximum(floor, _floor)
    intersect_botright = tf.minimum(ceil , _ceil)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.mul(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])
    # calculate the IOUs
    iou = tf.truediv(intersect, _area + area - intersect)
    return iou

def getImagelist(listfile):
    with open(listfile,'rU') as f:
      imagepath = [word.strip('\n') for word in f]
      tmppath = [labe.replace('JPEGImages','labels') for labe in imagepath]
      labepath = [labe.replace('jpg','txt') for labe in tmppath]
      datacount = len(imagepath)
      return imagepath,labepath,datacount
imagepath,labepath,datacount = getImagelist(train_file1)
test_imagepath, test_labepath, test_datacount = getImagelist(test_file)
print "all dataset count:",datacount

def constrain(mindata,maxdata,data):
    if data<mindata:
        data = mindata
    if data>maxdata:
        data = maxdata
    return data

def makelabeldata(labelpath,sx,sy,dx,dy,flip):
    probs = np.zeros([H*W,B,C])
    confs = np.zeros([H*W,B])
    coord = np.zeros([H*W,B,4])
    proid = np.zeros([H*W,B,C])
    prear = np.zeros([H*W,4])
    
    #read
    with open(labelpath,'rU') as f:
        labelpatch = [word.strip('\n').split() for word in f if word!='']
    objectNum = 0
    shape = [30, H*W, B, 4]
    global_coord = np.zeros(shape)

    for NumObj in xrange(len(labelpatch)):
        types = int(labelpatch[NumObj][0])
        # print types
        x = float(labelpatch[NumObj][1])
        y = float(labelpatch[NumObj][2])
        w = float(labelpatch[NumObj][3])
        h = float(labelpatch[NumObj][4])
        left   = x - w/2
        right  = x + w/2
        top    = y - h/2
        bottom = y + h/2
    #correct
        left   = left  * sx - dx
        right  = right * sx - dx
        top    = top   * sy - dy
        bottom = bottom* sy - dy

        if flip :
            left, right = 1. - right, 1. - left

        left =  constrain(0, 1, left)
        right = constrain(0, 1, right)
        top =   constrain(0, 1, top)
        bottom =constrain(0, 1, bottom)

        x = (left + right)/2.0
        y = (top + bottom)/2.0
        w = right -left
        h = bottom - top
        w = constrain(0, 1, w)
        h = constrain(0, 1, h)
    #label

        if (w < .01 or h < .01) :
            continue
        col = int(x * side_)
        row = int(y * side_)
        x_cell = x*side_ - col
        y_cell = y*side_ - row
        index = W * row + col

        probs[index, :, :] = [[0.]*C] * B
        probs[index, :, types] = 1.
        proid[index, :, :] = [[1.]*C] * B
        coord[index, :, :] = [[x_cell, y_cell, w, h]] * B
        prear[index, 0] = 0 - w * .5   # xleft
        prear[index, 1] = 0 - h * .5   # yup
        prear[index, 2] = 0 + w * .5  # xright
        prear[index, 3] = 0 + h * .5   # ybot
        confs[index, :] = [1.] * B
        global_coord[objectNum,...] = [left, top, right, bottom]
        objectNum = objectNum+1
        if objectNum >= 30:
            break

    # Finalise the placeholders' values
    upleft   = np.expand_dims(prear[:,0:2], 1)
    botright = np.expand_dims(prear[:,2:4], 1)
    wh = botright - upleft
    area = wh[:,:,0] * wh[:,:,1]
    upleft   = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)
    global_upleft = global_coord[..., 0:2]
    global_botright = global_coord[..., 2:4]
    global_wh = global_botright - global_upleft
    global_areas = global_wh[..., 0] * global_wh[..., 1]

    
    # value for placeholder at loss layer
    loss_feed_val = {
        'probs': probs, 'confs': confs, 
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft, 
        'botright': botright,
        'global_upleft': global_upleft, 'global_botright': global_botright,
        'global_wh': global_wh, 'global_areas': global_areas
    }
    return loss_feed_val
'''
def makelabeldata(labelpath,sx,sy,dx,dy):
    probs = np.zeros([side_, side_, num, classes_])
    confs = np.zeros([side_, side_, num])
    coord = np.zeros([side_, side_, num, 4])
    prear = np.zeros([side_, side_, 4])
    
    #read
    with open(labelpath,'rU') as f:
        labelpatch = [word.strip('\n').split() for word in f if word!='']
    objectNum = 0
    shape = [30, side_, side_, num, 4]
    global_coord = np.zeros(shape)

    for NumObj in xrange(len(labelpatch)):
        types = int(labelpatch[NumObj][0])
        # print types
        x = float(labelpatch[NumObj][1])
        y = float(labelpatch[NumObj][2])
        w = float(labelpatch[NumObj][3])
        h = float(labelpatch[NumObj][4])
        left   = x - w/2
        right  = x + w/2
        top    = y - h/2
        bottom = y + h/2
    #correct
        left   = left  * sx - dx
        right  = right * sx - dx
        top    = top   * sy - dy
        bottom = bottom* sy - dy
        left =  constrain(0, 1, left)
        right = constrain(0, 1, right)
        top =   constrain(0, 1, top)
        bottom =constrain(0, 1, bottom)

        x = (left + right)/2.0
        y = (top + bottom)/2.0
        w = right -left
        h = bottom - top
        w = constrain(0, 1, w)
        h = constrain(0, 1, h)
    #label

        if (w < .01 or h < .01) :
            continue
        col = int(x * side_)
        row = int(y * side_)
        x_cell = x*side_ - col
        y_cell = y*side_ - row

        probs[row, col, :, :] = [[0.]*classes_] * num
        probs[row, col, :, types] =[1.] * num
        confs[row, col,:] = [1.] * num
        coord[row, col, :, :] = [[x_cell,y_cell,np.log(w*side_/anchors[2*n]),np.log(h*side_/anchors[2*n+1])] for n in range(5)]
        # for the sake of truth_shift.x = 0 truth_shift.y = 0;
        prear[row, col, :] = [-w/2, -h/2, w/2, h/2]
        
        objectNum = objectNum+1
        if objectNum>30:
            break
        #for best_iou > l.thresh l.delta[index + 4] = 0;
        global_coord[objectNum-1,...] = [left, top, right, bottom]
    upleft = np.expand_dims(prear[:, :, 0:2], 2)
    botright = np.expand_dims(prear[:, :, 2:4], 2)
    wh = botright - upleft
    area = wh[:, :, :, 0] * wh[:, :, :, 1]
    upleft   = np.concatenate([upleft] * num, 2)
    botright = np.concatenate([botright] * num, 2)
    areas = np.concatenate([area] * num, 2)
    global_upleft = global_coord[..., 0:2]
    global_botright = global_coord[..., 2:4]
    global_wh = global_botright - global_upleft
    global_areas = global_wh[..., 0] * global_wh[..., 1]
    loss_feed_val = {
        'probs': probs, 'confs': confs,
        'coord': coord,
        'areas': areas, 'upleft': upleft,
        'botright': botright, 'global_coord': global_coord,
        'global_upleft': global_upleft, 'global_botright': global_botright,
        'global_wh': global_wh, 'global_areas': global_areas
    }
    return loss_feed_val
'''
def testimageload():
    dataset = np.ndarray(shape=(batch_size, input_size_, input_size_, 3), dtype=np.float32)
    ResizeSize = (input_size_, input_size_)
    blist_webId = random.sample(range(0,test_datacount), batch_size)
    feed_batch = dict()
    for i in range(0, batch_size):
        orig = cv2.imread(test_imagepath[blist_webId[i]])
        orig = cv2.resize(orig,ResizeSize)
        orig = orig / 255.0
        dataset[i,:,:,:] = orig
        new_feed = makelabeldata(test_labepath[blist_webId[i]],1.0,1.0,0.0,0.0)
        for key in new_feed:
            new = new_feed[key]
            old_feed = feed_batch.get(key, np.zeros((0,) + new.shape))
            feed_batch[key] = np.concatenate([old_feed, [new]])
    return dataset, feed_batch

def imageloadOne(shuffle_idx, batch_index):
    dataset = np.ndarray(shape=(batch_size, input_size_, input_size_, 3), dtype=np.float32)
    labels = np.ndarray(shape=(batch_size,BBNum*(classes_+1+coord_)), dtype=np.float32)
    ResizeSize = (input_size_, input_size_)
    feed_batch = dict()
    for i in range(batch_size*batch_index, batch_size*batch_index + batch_size):
        orig = cv2.imread(imagepath[shuffle_idx[i]])
        orig = cv2.resize(orig,ResizeSize)
        orig = orig / 255.0
        #orig = orig * 2.0 -1.0
        flip = np.random.randint(0, 2)
        if flip :
            orig = cv2.flip(orig, 1)
        dataset[i-batch_size*batch_index,:,:,:] = orig
        new_feed = makelabeldata(labepath[shuffle_idx[i]],1.0,1.0,0.0,0.0,flip)
        for key in new_feed:
            new = new_feed[key]
            old_feed = feed_batch.get(key, np.zeros((0,) + new.shape))
            feed_batch[key] = np.concatenate([old_feed, [new]])
    return dataset, feed_batch


def imageloadRandom(shuffle_idx, batch_index):
      dataset = np.ndarray(shape=(batch_size, input_size_, input_size_, 3), dtype=np.float32)
      labels = np.ndarray(shape=(batch_size,BBNum*(classes_+1+coord_)), dtype=np.float32)
      ResizeSize = (input_size_, input_size_)
      feed_batch = dict() 
      for i in range(batch_size*batch_index, batch_size*batch_index + batch_size):
          orig = cv2.imread(imagepath[shuffle_idx[i]])
          # print imagepath[0]
          #orig = cv2.resize(orig,ResizeSize)
          orig = random_distort_image(orig, 0.1, 1.5, 1.5)
          oh = orig.shape[0]
          ow = orig.shape[1]
          jitter = 0.2
          dw = (ow*jitter)
          dh = (oh*jitter)
          pleft  = int(np.random.uniform(-dw, dw))
          pright = int(np.random.uniform(-dw, dw))
          ptop   =  int(np.random.uniform(-dh, dh))
          pbot   =  int(np.random.uniform(-dh, dh))
          swidth =  ow - pleft - pright
          sheight = oh - ptop - pbot
          sx = 1.0*swidth /ow
          sy = 1.0*sheight/oh
          #croppedImage = crop_image(orig, pleft, ptop, swidth, sheight)
          x_start = max(pleft,0)
          x_end  = min(ow-pright,ow)
          y_start = max(ptop,0)
          y_end = min(oh-pbot,oh)
          crop_img = orig[y_start:y_end, x_start:x_end]

          pad_top = max(0,-ptop)
          pad_botton = max(0,-pbot)
          pad_left=max(0,-pleft)
          pad_right = max(0,-pright)
          crop_img = crop_img / 255.0 #* random.uniform(0.7, 1.2)
          crop_img = cv2.copyMakeBorder(crop_img, pad_top,pad_botton,pad_left,pad_right, cv2.BORDER_CONSTANT, value=0)
          crop_img = cv2.resize(crop_img,ResizeSize)
          flip = np.random.randint(0, 2)
          if flip :
              crop_img = cv2.flip(crop_img, 1)
          dataset[i-batch_size*batch_index,:,:,:] = crop_img
          dx = 1.0*pleft/ow/sx
          dy = 1.0*ptop /oh/sy
          new_feed = makelabeldata(labepath[shuffle_idx[i]],1.0/sx,1.0/sy,dx,dy,flip)
          for key in new_feed:
              new = new_feed[key]
              old_feed = feed_batch.get(key, np.zeros((0,) + new.shape))
              feed_batch[key] = np.concatenate([old_feed, [new]])
      return dataset, feed_batch

def  overlap( x1,  w1,  x2,  w2):
     l1 = x1 - w1/2
     l2 = x2 - w2/2
     left = max(l1,l2)
     r1 = x1 + w1/2
     r2 = x2 + w2/2
     right = min(r1,r2)
     return right - left
# x,y,w,h :0,1,2,3
def box_iou(a,b):
   w = overlap(a[0], a[2], b[0], b[2])
   h = overlap(a[1], a[3], b[1], b[3])
   if(w < 0 or h < 0):
       return 0
   inter = w*h
   union = a[2]*a[3] + b[2]*b[3] - inter
   return inter/union


def do_nms_sort(boxes, probs):
    s = []
    total =BBNum*num
    for i in range(0,total):
        s.append((i,probs[i]))

    for k in range(0,classes_):
        s = sorted(s,reverse=True,key=lambda x:x[1][k])
        for i in range(0,total):
            if probs[s[i][0]][k]==0:
                continue
            boxA = boxes[s[i][0]]
            for j in range(i+1,total):
                boxB = boxes[s[j][0]]
                if probs[s[j][0]][k]==0 :
                    continue
                if box_iou(boxA,boxB)>iouThresh:
                    probs[s[j][0]][k] = 0
                    #print "ignore BB ",i
    return probs
weight_list = []

class YOLO_TF:
    fromfile = None
    tofile_img = 'test/output.jpg'
    tofile_txt = 'test/output.txt'
    imshow = True
    filewrite_img = False
    filewrite_txt = False
    disp_console = True
    weights_file = 'weight/Record_2E-YOLO_small_43'
    alpha = 0.1
    threshold = 0.05
    iou_threshold = 0.5
    classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    test = False
    train = False
    testimage = False
    w_img = 640
    h_img = 480
    all_anchors = None
    anchors = np.array([[0, 0, 1.08, 1.19], [0, 0, 3.42, 4.41], [0, 0, 6.63, 11.38], [0, 0, 9.42, 5.11], [0, 0, 16.62, 10.52]], dtype=np.float32)
    object_scale = 5.0
    noobject_scale = 1.0
    class_scale = 1.0
    coord_scale = 1.0

    def __init__(self,argvs = []):
        self.argv_parser(argvs)
        self.build_networks()
        if self.train is not False: self.training()
        if self.fromfile is not None: self.detect_from_file(self.fromfile)
        if self.test is not False: self.testAP()
        
    def argv_parser(self,argvs):
        for i in range(1,len(argvs),2):
            if argvs[i] == '-fromfile' : self.fromfile = argvs[i+1]
            if argvs[i] == '-tofile_img' : self.tofile_img = argvs[i+1] ; self.filewrite_img = True
            if argvs[i] == '-tofile_txt' : self.tofile_txt = argvs[i+1] ; self.filewrite_txt = True
            if argvs[i] == '-imshow' :
                if argvs[i+1] == '1' :self.imshow = True
                else : self.imshow = False
            if argvs[i] == '-train' : self.train = True
            if argvs[i] == '-test' : self.test = True
            if argvs[i] == '-image' : self.testimage = True
            if argvs[i] == '-disp_console' :
                if argvs[i+1] == '1' :self.disp_console = True
                else : self.disp_console = False
                
    def build_networks(self):
      #with tf.device('/cpu:0'):
        if self.disp_console : print "Building YOLO_v2_tiny graph..."
        self.image = tf.placeholder('float32',[None,416,416,3])
        self.train_phase = tf.placeholder(tf.bool)

        #weight_list = []

        self.conv_1 , weight = self.conv_layer_bn("conv1",self.image,16,3,1, wd=0.0005)
        weight_list.append(weight)
        self.pool_1 = self.pooling_layer("pool1",self.conv_1,2,2)
        self.conv_2 , weight = self.conv_layer_bn("conv2",self.pool_1,32,3,1, wd=0.0005)
        weight_list.append(weight)
        self.pool_2 = self.pooling_layer("pool2",self.conv_2,2,2)
        self.conv_3 , weight = self.conv_layer_bn("conv3",self.pool_2,64,3,1, wd=0.0005)
        weight_list.append(weight)
        self.pool_3 = self.pooling_layer("pool3",self.conv_3,2,2)
        self.conv_4 , weight = self.conv_layer_bn("conv4",self.pool_3,128,3,1, wd=0.0005)
        weight_list.append(weight)
        self.pool_4 = self.pooling_layer("pool4",self.conv_4,2,2)
        self.conv_5 , weight = self.conv_layer_bn("conv5",self.pool_4,256,3,1, wd=0.0005)
        weight_list.append(weight)
        self.pool_5 = self.pooling_layer("pool5",self.conv_5,2,2)
        self.conv_6, weight  = self.conv_layer_bn("conv6",self.pool_5,512,3,1, wd=0.0005)
        weight_list.append(weight)
        self.pool_6 = self.pooling_layer("pool6",self.conv_6,2,1)
        self.conv_7, weight = self.conv_layer_bn("conv7",self.pool_6,1024,3,1, wd=0.0005)
        weight_list.append(weight)
        self.conv_8, weight  = self.conv_layer_bn("conv8",self.conv_7,1024,3,1, wd=0.0005)
        weight_list.append(weight)
        self.conv_9, weight, biases = self.conv_layer("conv9",self.conv_8,125,1,1,linear=True, wd=0.0005)
        weight_list.append(weight)
        weight_list.append(biases)
        self.yolo_loss, self.best_box  = self.yololoss(self.conv_9)
        tf.add_to_collection('losses',self.yolo_loss)
        self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        self.yolopred = self.yoloout(self.conv_9)

        #self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.global_step = tf.Variable(0,name='global_step',trainable=False)
        decay_steps = 10000
        decay_rate = 0.9
        learning_rate = 0.0001
        self.lr = tf.train.exponential_decay(learning_rate, self.global_step, decay_steps, decay_rate, staircase=True)
        self.opt = tf.train.AdamOptimizer(0.0001)
        self.grads = self.opt.compute_gradients(self.loss)
        self.apply_gradient_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)
        self.train_op = self.apply_gradient_op

        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.merged_summary_op = tf.summary.merge_all()

        # Create a saver.
        self.saver = tf.train.Saver(tf.all_variables())
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
       
        #if self.is_training: 
        #    self.train_writer = tf.summary.FileWriter('log', self.sess.graph)
        self.sess.run(init)

        self.checkpoint = tf.train.get_checkpoint_state("train_weight")
        if self.checkpoint and self.checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, self.checkpoint.model_checkpoint_path)
            print "Successfully loaded:", self.checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

        if self.disp_console : print "Loading test net complete!" + '\n'

    def generate_anchors(self, anchors):
        zeros_array = np.zeros((batch_size, side_, side_, num, 4), dtype=np.float32)
        self.all_anchors = zeros_array + anchors

    def yoloout(self, predictions):
        predictions = tf.reshape(predictions, [1, side_, side_, num, -1])
        pre_obj = predictions[:, :, :, :, 4]
        pre_obj = tf.sigmoid(pre_obj)
        pre_obj = tf.expand_dims(pre_obj, dim=4)
        pre_coord = predictions[:, :, :, :, 0:4]
        pre_class = predictions[:, :, : ,:, 5:25]
        pre_class = tf.reshape(pre_class, [-1, 20])
        pre_class = tf.nn.softmax(pre_class)
        pre_class = tf.reshape(pre_class, [1, side_, side_, num, 20])
        pre_out = tf.concat(4, [pre_coord, pre_obj, pre_class])
        pre_out = tf.reshape(pre_out, [1, side_, side_, num*25])
        return pre_out

    def yololoss(self, predictions):
        size1 = [None, HW, B, C]
        size2 = [None, HW, B]
        size3 = [None, 30, HW, B]

        # return the below placeholders
        _probs = tf.placeholder(tf.float32, size1)
        _confs = tf.placeholder(tf.float32, size2)
        _coord = tf.placeholder(tf.float32, size2 + [4])
         
        # weights term for L2 loss
        _proid = tf.placeholder(tf.float32, size1)
        # material calculating IOU
        _areas = tf.placeholder(tf.float32, size2)
        _upleft = tf.placeholder(tf.float32, size2 + [2])
        _botright = tf.placeholder(tf.float32, size2 + [2])

        _global_upleft = tf.placeholder(tf.float32, size3 + [2]) 
        _global_botright = tf.placeholder(tf.float32, size3 + [2])
        _global_wh = tf.placeholder(tf.float32, size3 + [2])
        _global_areas = tf.placeholder(tf.float32, size3)

       
        self.placeholders = {
        'probs':_probs, 'confs':_confs, 'coord':_coord, 'proid':_proid,
        'areas':_areas, 'upleft':_upleft, 'botright':_botright,
        'global_upleft':_global_upleft, 'global_botright':_global_botright, 
        'global_wh':_global_wh, 'global_areas':_global_areas
        }
        
        # Extract the coordinate prediction from predictions
        net_out_reshape = tf.reshape(predictions, [-1, H, W, B, (4 + 1 + C)])
        coords = net_out_reshape[:, :, :, :, :4]
        coords = tf.reshape(coords, [-1, H*W, B, 4])
        adjusted_coords_xy = expit_tensor(coords[:,:,:,0:2])
        adjusted_coords_wh = tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2])
        coords = tf.concat(3, [adjusted_coords_xy, adjusted_coords_wh])

        adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4])
        adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1])

        adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
        adjusted_prob = tf.reshape(adjusted_prob, [-1, H*W, B, C])

        adjusted_net_out = tf.concat(3, [adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob])        

        wh = np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2])
        wh = tf.constant(wh, dtype=tf.float32)
        #wh = tf.pow(coords[:,:,:,2:4], 2) *  np.reshape([W, H], [1, 1, 1, 2])
        area_pred = wh[:,:,:,0] * wh[:,:,:,1]
        #centers = coords[:,:,:,0:2]
        centers = tf.zeros([1, 1, B, 2])
        floor = centers - (wh * .5)
        ceil  = centers + (wh * .5)

        # calculate the intersection areas
        intersect_upleft   = tf.maximum(floor, _upleft)
        intersect_botright = tf.minimum(ceil , _botright)
        intersect_wh = intersect_botright - intersect_upleft
        intersect_wh = tf.maximum(intersect_wh, 0.0)
        intersect = tf.mul(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

        # calculate the best IOU, set 0.0 confidence for worse boxes
        iou = tf.truediv(intersect, _areas + area_pred - intersect)
        best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
        best_box = tf.to_float(best_box)
        confs = tf.mul(best_box, _confs)

        # calculate the best IOU for every cell 
        wh_pre = coords[:, :, :, 2:4]
        area_pre = wh_pre[:, :, :, 0] * wh_pre[:, :, :, 1]
        centers_pre = coords[:, :, :, 0:2]
        transform_array = tf.reshape(tf.expand_dims(tf.constant(generate_posarray(), dtype=tf.float32), dim=2), [HW,1,2])
        centers_pre = (centers_pre + transform_array) / 13.
        floor_pre = centers_pre - (wh_pre * .5)
        ceil_pre = centers_pre + (wh_pre * .5)

        area_pre = tf.expand_dims(area_pre, axis = 1)
        centers_pre = tf.expand_dims(centers_pre, axis = 1)
        floor_pre = tf.expand_dims(floor_pre, axis = 1)
        ceil_pre = tf.expand_dims(ceil_pre, axis = 1)
        intersect_upleft_pre = tf.maximum(floor_pre, _global_upleft)
        intersect_botright_pre = tf.minimum(ceil_pre, _global_botright)
        intersect_wh_pre = intersect_botright_pre - intersect_upleft_pre
        intersect_wh_pre = tf.maximum(intersect_wh_pre, 0.0)
        intersect_pre = tf.mul(intersect_wh_pre[:, :, :, :, 0], intersect_wh_pre[:, :, :, :, 1])
        iou_pre = tf.truediv(intersect_pre, _global_areas + area_pre - intersect_pre)
        best_box_pre_tmp = tf.to_float(tf.equal(iou_pre, tf.reduce_max(iou_pre,[1], True)))
        thresh_mask = tf.to_float(tf.greater(iou_pre, 0.6))
        best_box_pre_tmp_ = tf.mul(best_box_pre_tmp, thresh_mask)
        best_box_pre = tf.reduce_sum(best_box_pre_tmp_, 1)

        # iou between pre and ground truth
        #iou_coord = coord_iou(coords, _coord)
        
        # take care of the weight terms
        conid = snoob * (1. - confs) * (1. - best_box_pre) + sconf * confs
        weight_coo = tf.concat(3, 4 * [tf.expand_dims(confs, -1)])
        cooid = scoor * weight_coo
        weight_pro = tf.concat(3, C * [tf.expand_dims(confs, -1)])
        proid = sprob * weight_pro 

        true = tf.concat(3, [_coord, tf.expand_dims(confs, 3), _probs ])
        wght = tf.concat(3, [cooid, tf.expand_dims(conid, 3), proid ])

        loss = tf.pow(adjusted_net_out - true, 2)
        loss = tf.mul(loss, wght)
        loss = tf.reshape(loss, [-1, H*W*B*(4 + 1 + C)])
        loss = tf.reduce_sum(loss, 1)
        loss = .5 * tf.reduce_mean(loss)
        return loss, best_box_pre

    '''
    def yololoss(self,predictions):
        size1 = [None, side_, side_, num, classes_]
        size2= [None, side_, side_, num]
        size3 = [None, 30,side_, side_, num]
        # return the below placeholders
        _probs = tf.placeholder(tf.float32, size1)
        _confs = tf.placeholder(tf.float32,size2 )
        _coord = tf.placeholder(tf.float32, size2 + [4])
        _areas = tf.placeholder(tf.float32, size2)
        _upleft = tf.placeholder(tf.float32, size2 + [2])
        _botright = tf.placeholder(tf.float32, size2 + [2])
        _global_coord = tf.placeholder(tf.float32, size3 + [4])
        _global_upleft = tf.placeholder(tf.float32, size3 + [2]) 
        _global_botright = tf.placeholder(tf.float32, size3 + [2])
        _global_wh = tf.placeholder(tf.float32, size3 + [2])
        _global_areas = tf.placeholder(tf.float32, size3)

        self.placeholders = {
            'probs':_probs, 'confs':_confs,
            'coord':_coord,
            'areas':_areas, 'upleft': _upleft,
            'botright':_botright, 'global_coord':_global_coord,
            'global_upleft':_global_upleft, 'global_botright':_global_botright,
            'global_wh':_global_wh, 'global_areas':_global_areas
        }
        

        

        predictions = tf.reshape(predictions, [batch_size, side_, side_, num, -1])
        self.pre_obj = predictions[:, :, :, :, 4] 
        self.pre_obj = tf.sigmoid(self.pre_obj)
        self.pre_xy = predictions[:, :, :, :, :2]
        self.pre_xy = tf.sigmoid(self.pre_xy)
        self.pre_wh = predictions[:, :, :, :, 2:4]
        self.pre_coord = tf.concat(4, [self.pre_xy, self.pre_wh])
        self.pre_class = predictions[:, :, : ,:, 5:25]
        self.pre_class = tf.nn.softmax(self.pre_class)

        # generate mask
        self.generate_anchors(self.anchors)
        wh = self.all_anchors[:, :, :, :, 2:4] / side_
        area = wh[:, :, :, :,0] * wh[:, :, :, :,1]
        centers = self.all_anchors[:, :, :, :, 0:2] / side_
        floor = centers - (wh * .5)
        ceil = centers + (wh * .5)

        # calculate the intersection areas
        intersect_upleft   = tf.maximum(floor, _upleft)
        intersect_botright = tf.minimum(ceil , _botright)
        intersect_wh = intersect_botright - intersect_upleft
        intersect_wh = tf.maximum(intersect_wh, 0.0)
        intersect = tf.mul(intersect_wh[:,:,:,:,0], intersect_wh[:,:,:,:, 1])
 
        # calculate the best IOU, set0.0 confidence for worse boxes
        iou = tf.truediv(intersect, _areas + area - intersect)
        best_box = tf.equal(iou, tf.reduce_max(iou, [3], True))
        best_box = tf.to_float(best_box)
        confs = tf.mul(best_box, _confs)

        # calculate the best IOU for each prediction which greater than an threshold
        transform_array = tf.expand_dims(tf.constant(generate_posarray(), dtype=tf.float32), dim=2)
        transform_array = tf.concat(2, num * [transform_array])
        self.pre_xy_tran =  tf.truediv(self.pre_xy + transform_array, 13.)
        self.pre_wh_tran =  tf.truediv(tf.mul(tf.exp(self.pre_wh), self.all_anchors[:, :, :, :, 2:4]), 13.)
        area_pred = self.pre_wh_tran[:, :, :, :, 0] * self.pre_wh_tran[:, :, :, :, 1]
        floor_pred = self.pre_xy_tran - (self.pre_wh_tran * .5)
        ceil_pred = self.pre_xy_tran + (self.pre_wh_tran * .5)
        floor_pred = tf.expand_dims(floor_pred, dim=1) 
        ceil_pred = tf.expand_dims(ceil_pred, dim=1)
        area_pred = tf.expand_dims(area_pred, dim=1)
        intersect_upleft_pre = tf.maximum(floor_pred, _global_upleft)
        intersect_botright_pre = tf.minimum(ceil_pred, _global_botright)
        intersect_wh_pre = intersect_botright_pre - intersect_upleft_pre
        intersect_wh_pre = tf.maximum(intersect_wh_pre, 0.0)
        intersect_pre = tf.mul(intersect_wh_pre[:, :, :, :, :, 0], intersect_wh_pre[:, :, :, :, :, 1])
        iou_pre = tf.truediv(intersect_pre, _global_areas + area_pred - intersect_pre + 0.000001)
        best_box_pre_tmp = tf.to_float(tf.equal(iou_pre, tf.reduce_max(iou_pre,[1], True)))
        thresh_mask = tf.to_float(tf.greater(iou_pre, 0.6))
        best_box_pre_tmp_ = tf.mul(best_box_pre_tmp, thresh_mask)
        best_box_pre = tf.reduce_sum(best_box_pre_tmp_, 1)
        
         

        con_idx = self.noobject_scale * (1 - confs) * (1 - best_box_pre) + self.object_scale * confs
        coord_idx = self.coord_scale * tf.concat(4, 4 * [tf.expand_dims(confs, -1)])
        pro_idx = self.class_scale * tf.concat(4, 20 * [tf.expand_dims(confs, -1)])

        #con_loss = tf.mul(tf.pow(self.pre_obj - confs, 2), con_idx)
        #self.con_loss_sum = tf.reduce_sum(con_loss, [1,2,3])
        #coord_loss = tf.mul(tf.pow(self.pre_coord - _coord, 2), coord_idx)
        #self.coord_loss_sum = tf.reduce_sum(coord_loss, [1,2,3,4])
        #class_loss = tf.mul(tf.pow(self.pre_class - _probs, 2), pro_idx) 
        #self.class_loss_sum = tf.reduce_sum(class_loss, [1,2,3,4])
        #loss = tf.reduce_mean(self.con_loss_sum + self.coord_loss_sum + self.class_loss_sum)

        con_loss = tf.mul(tf.pow(self.pre_obj - confs, 2), con_idx)
        con_loss_sum = tf.reduce_sum(con_loss, [1,2,3])
        self.con_loss = tf.reduce_mean(con_loss_sum)
        coord_loss = tf.mul(tf.pow(self.pre_coord - _coord, 2), coord_idx)
        coord_loss_sum = tf.reduce_sum(coord_loss, [1,2,3,4])
        self.coord_loss = tf.reduce_mean(coord_loss_sum)
        class_loss = tf.mul(tf.pow(self.pre_class - _probs, 2), pro_idx)
        class_loss_sum = tf.reduce_sum(class_loss, [1,2,3,4])
        self.class_loss = tf.reduce_mean(class_loss_sum)
        loss = self.con_loss + self.coord_loss + self.class_loss
        return loss
    '''

    def variable(self, name, shape=None, dtype=tf.float32, initializer=None,
                 regularizer=None, trainable=True, collections=None, device='',
                 restore=True):
        """Gets an existing variable with these parameters or creates a new one.
        It also add itself to a group with its name.
        Args:
            name: the name of the new or existing variable.
            shape: shape of the new or existing variable.
            dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
            initializer: initializer for the variable if one is created.
            regularizer: a (Tensor -> Tensor or None) function; the result of
               applying it on a newly created variable will be added to the collection
               GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
            trainable: If `True` also add the variable to the graph collection
              `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
            collections: A list of collection names to which the Variable will be added.
              Note that the variable is always also added to the tf.GraphKeys.VARIABLES
              and MODEL_VARIABLES collections.
            device: Optional device to place the variable. It can be an string or a
              function that is called to get the device for the variable.
            restore: whether the variable should be added to the
              VARIABLES_TO_RESTORE collection.
        Returns:
          The created or existing variable.
        """
        collections = list(collections or [])

        # Make sure variables are added to tf.GraphKeys.VARIABLES and MODEL_VARIABLES
        collections += [tf.GraphKeys.VARIABLES, MODEL_VARIABLES]
        # Add to VARIABLES_TO_RESTORE if necessary
        if restore:
            collections.append(VARIABLES_TO_RESTORE)
        # Remove duplicates
        collections = set(collections)
        # Get the device for the variable.
        return tf.get_variable(name, shape=shape, dtype=dtype,
                               initializer=initializer, regularizer=regularizer,
                               trainable=trainable, collections=collections)

    def batch_norm_layer(self, x,train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=False, scale=True,
        updates_collections=None,
        is_training=True,
        reuse=None,
        variables_collections= UPDATE_OPS_COLLECTION,
        trainable=True,
        scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=False, scale=True,
        updates_collections=None,
        is_training=False,
        reuse=True,
        variables_collections= UPDATE_OPS_COLLECTION,
        trainable=True,
        scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


    def batch_norm_delete(self,
                   inputs,
                   decay=0.999,
                   center=True,
                   scale=True,
                   epsilon=0.001,
                   moving_vars='moving_vars',
                   activation=None,
                   is_training=True,
                   trainable=True,
                   restore=True,
                   scope=None,
                   reuse=None):
        """Adds a Batch Normalization layer.
        Args:
          inputs: a tensor of size [batch_size, height, width, channels]
                  or [batch_size, channels].
          decay: decay for the moving average.
          center: If True, subtract beta. If False, beta is not created and ignored.
          scale: If True, multiply by gamma. If False, gamma is
            not used. When the next layer is linear (also e.g. ReLU), this can be
            disabled since the scaling can be done by the next layer.
          epsilon: small float added to variance to avoid dividing by zero.
          moving_vars: collection to store the moving_mean and moving_variance.
          activation: activation function.
          is_training: whether or not the model is in training mode.
          trainable: whether or not the variables should be trainable or not.
          restore: whether or not the variables should be marked for restore.
          scope: Optional scope for variable_op_scope.
          reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        Returns:
          a tensor representing the output of the operation.
        """
        inputs_shape = inputs.get_shape()
        axis = list(range(len(inputs_shape) - 1))
        params_shape = inputs_shape[-1:]
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta = self.variable('beta',
                                 params_shape,
                                 initializer=tf.zeros_initializer(),
                                 trainable=trainable,
                                 restore=restore)
        if scale:
            gamma = self.variable('gamma',
                                  params_shape,
                                  initializer=tf.ones_initializer(),
                                  trainable=trainable,
                                  restore=restore)
        # Create moving_mean and moving_variance add them to
        # GraphKeys.MOVING_AVERAGE_VARIABLES collections.
        moving_collections = [moving_vars, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
        moving_mean = self.variable('moving_mean',
                                    params_shape,
                                    initializer=tf.zeros_initializer(),
                                    trainable=False,
                                    restore=restore,
                                    collections=moving_collections)
        moving_variance = self.variable('moving_variance',
                                        params_shape,
                                        initializer=tf.ones_initializer(),
                                        trainable=False,
                                        restore=restore,
                                        collections=moving_collections)

        def if_true():
            mean, variance = tf.nn.moments(inputs, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
            return mean, variance

        def if_false():
            mean = moving_mean
            variance = moving_variance
            return mean, variance

        mean, variance = tf.cond(is_training, if_true, if_false)
        '''
        if is_training:
            # Calculate the moments based on the individual batch.
            mean, variance = tf.nn.moments(inputs, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
        else:
            # Just use the moving_mean and moving_variance.
            mean = moving_mean
            variance = moving_variance
        '''
        # Normalize the activations.
        outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(inputs.get_shape()) 
        if activation:
            outputs = activation(outputs)
        return outputs

    def conv_layer_bn(self,name,inputs,filters,size,stride,linear=False, wd=0.0):
        with tf.variable_scope(name):
            input_h,input_w,channels = inputs.get_shape().as_list()[1:]
            scale = tf.sqrt(2.0/ (size*size*channels))
            weight = tf.get_variable("weights", shape=[size, size, channels, filters], initializer=tf.random_uniform_initializer(-scale, scale))
            biases = tf.get_variable("biases", shape=[filters], initializer=tf.constant_initializer(0.0))

            pad_size = size//2
            pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
            inputs_pad = tf.pad(inputs,pad_mat)

            conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID')
            conv_bn = self.batch_norm_layer(conv, self.train_phase, "batch_norm")    
            conv_biased = tf.add(conv_bn,biases)

            if wd :
                weight_decay =  tf.mul(tf.nn.l2_loss(weight), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)

            tf.summary.histogram("weights", weight)
            tf.summary.histogram("biases", biases)
            output_h, output_w, output_channels = conv_biased.get_shape()[1:]    
            if self.disp_console : print '    Layer  {} : size={}x{}/{}, input = {}x{}x{} -> output = {}x{}x{}'.format(name,size,size,stride, input_w, input_h, channels, output_w, output_h, output_channels)
            if linear : return conv_biased, weight
            return tf.maximum(self.alpha*conv_biased,conv_biased), weight

    def conv_layer1(self,idx,inputs,filters,size,stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.03))
        biases = tf.Variable(tf.constant(0.03, shape=[filters]))

        pad_size = size//2
        pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
        inputs_pad = tf.pad(inputs,pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')    
        conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')    
        if self.disp_console : print '    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx,size,size,stride,filters,int(channels))
        return tf.maximum(self.alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu'), weight, biases

    def conv_layer(self,name,inputs,filters,size,stride,linear=False, wd=0.0):
        with tf.variable_scope(name):
            input_h,input_w,channels = inputs.get_shape().as_list()[1:]
            scale = tf.sqrt(2.0/ (size*size*channels))
            weight = tf.get_variable("weights", shape=[size, size, channels, filters], initializer=tf.random_uniform_initializer(-scale, scale))
            biases = tf.get_variable("biases", shape=[filters], initializer=tf.constant_initializer(0.0))

            pad_size = size//2
            pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
            inputs_pad = tf.pad(inputs,pad_mat)

            conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID')
            conv_biased = tf.add(conv,biases)

            if wd :
                weight_decay =  tf.mul(tf.nn.l2_loss(weight), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)


            tf.summary.histogram("weights", weight)
            tf.summary.histogram("biases", biases)
            output_h, output_w, output_channels = conv_biased.get_shape()[1:]    
            if self.disp_console : print '    Layer  {} : size={}x{}/{}, input = {}x{}x{} -> output = {}x{}x{}'.format(name,size,size,stride, input_w, input_h, channels, output_w, output_h, output_channels)
            if linear : return conv_biased, weight, biases
            return tf.maximum(self.alpha*conv_biased,conv_biased), weight, biases

    def pooling_layer(self,name,inputs,size,stride):
        with tf.variable_scope(name):
            input_h,input_w,channels = inputs.get_shape()[1:]
            pooled = tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME')
            output_h, output_w, output_channels = pooled.get_shape()[1:]
            if self.disp_console : print '    Layer  {} : size={}x{}/{}, input = {}x{}x{} -> output = {}x{}x{}'.format(name,size,size,stride, input_w, input_h, channels, output_w, output_h, output_channels)
            return pooled


    def detect_from_cvmat(self,img):
        s = time.time()
        self.h_img,self.w_img,_ = img.shape
        img_resized = cv2.resize(img, (416, 416))
        #img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray(img_resized)
        inputs = np.zeros((1,416,416,3),dtype='float32')
        inputs[0] = img_resized_np/255.0
        #inputs[0] = img_resized
        in_dict = {self.image: inputs, self.train_phase : False}
        moving_mean = tf.get_collection(UPDATE_OPS_COLLECTION)
        net_output,conv_9 = self.sess.run([self.yolopred, self.conv_9], feed_dict=in_dict)
        strtime = str(time.time()-s)
        print net_output[0,7,6,79:100]
        probs, boxes = self.get_region_boxes(net_output[0])
        probs = do_nms_sort(boxes, probs)
        for j in range(0,BBNum*num):
            cls =np.argmax(probs[j])
            prob = probs[j][cls]
            if prob<thresh:
                continue
            left  = int((boxes[j][0] - (boxes[j][2]/2.)) * self.w_img)
            right = int((boxes[j][0]+(boxes[j][2]/2.)) * self.w_img)
            top   = int((boxes[j][1]-(boxes[j][3]/2.)) * self.h_img)
            bot   = int((boxes[j][1]+(boxes[j][3]/2.)) * self.h_img)
            print cls,prob,left,top,right,bot
            if(left < 0):
                left = 0
            if(right > self.w_img - 1):
                right = self.w_img - 1
            if(top < 0):
                top = 0
            if(bot > self.h_img - 1):
                bot = self.h_img -1
            lefttop = (left,top)
            rightbottom=(right,bot)
            cv2.rectangle(img, lefttop, rightbottom, (0,255,0), 2)
            #cv2.rectangle(img, (left,top-20), (int(0.3*right),top), (125,125,125), -1)
            cv2.putText(img, self.classes[cls] + ' : %.2f' % prob,(left+5,top-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            
        #strtime = str(time.time()-s)
        cv2.imshow('YOLO_v2 detection',img)
        cv2.waitKey(0)
        if self.disp_console : print 'Elapsed time : ' + strtime + ' secs' + '\n'

    def detect_from_file(self,filename):
        if self.disp_console : print 'Detect from ' + filename
        img = cv2.imread(filename)
        #img = misc.imread(filename)
        self.detect_from_cvmat(img)

    def get_region_boxes(self,output):
        probs = np.zeros(shape=(side_, side_, num, classes_), dtype=np.float32)
        boxes = np.zeros(shape=(side_, side_,num, 4), dtype=np.float32)
        for i in range(0,BBNum):
            row = i / side_
            col = i % side_
            for n in range(0, num):
                scale = output[row, col, n*25+4]
                boxes[row, col, n, 0] = (col + logistic_activate(output[row, col, n*25])) / side_
                boxes[row, col, n, 1] = (row + logistic_activate(output[row, col, n*25+1])) / side_ 
                boxes[row, col, n, 2] = (np.exp(output[row, col, n*25+2])) * anchors[2*n] / side_
                boxes[row, col, n, 3] = (np.exp(output[row, col, n*25+3])) * anchors[2*n+1] / side_
                for j in range(0,classes_):
                    prob =  scale * output[row, col, n*25+5+j]
                    if prob > thresh:
                        probs[row, col, n, j] = prob
        return probs.reshape((BBNum*num, classes_)), boxes.reshape((BBNum*num, 4))

    def show_results(self,img,results):
        img_cp = img.copy()
        if self.filewrite_txt :
            ftxt = open(self.tofile_txt,'w')
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3])//2
            h = int(results[i][4])//2
            if self.disp_console : print '    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + str(int(results[i][3])) + ',' + str(int(results[i][4]))+'], Confidence = ' + str(results[i][5])
            if self.filewrite_img or self.imshow:
                cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
                cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            if self.filewrite_txt :                
                ftxt.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h)+',' + str(results[i][5]) + '\n')
        if self.filewrite_img : 
            if self.disp_console : print '    image file writed : ' + self.tofile_img
            cv2.imwrite(self.tofile_img,img_cp)            
        if self.imshow :
            cv2.imshow('YOLO_small detection',img_cp)
            cv2.waitKey(1)
        if self.filewrite_txt : 
            if self.disp_console : print '    txt file writed : ' + self.tofile_txt
            ftxt.close()

    def iou(self,box1,box2):
        tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

    def training(self): #TODO add training function!
        loss_ph = self.placeholders
        train_writer = tf.summary.FileWriter('log/train', self.sess.graph)
        test_writer = tf.summary.FileWriter('log/test')

        shuffle_idx = np.random.permutation(np.arange(datacount))
        batch_index = 0
        for i in range(800000):
            if (batch_index * batch_size + batch_size) > datacount :
                #print batch_index
                shuffle_idx = np.random.permutation(np.arange(datacount))
                batch_index = 0
            dataset, datum  = imageloadRandom(shuffle_idx, batch_index)
            in_dict = {loss_ph[key]: datum[key] for key in loss_ph}
            in_dict[self.image] = dataset
            in_dict[self.train_phase] = True
            _, loss, lr, best_box = self.sess.run([self.train_op, self.yolo_loss, self.lr, self.best_box],feed_dict=in_dict)
            # print "count: %d coord loss : %f\n"%(countobj,coord)
            if i % 10 == 0:
                print "Iter:%d Lr:%f Loss:%f"%(i, lr,loss)
                #print best_box[0,:,:]
                #f = open("out.txt", "w")
                #print >>f, loss1
                print '\n'
                #print bestbox, '\n'
            if i % 100 == 0:
                summary = self.sess.run(self.merged_summary_op, feed_dict=in_dict)
                train_writer.add_summary(summary, global_step=i)
                
                #test_dataset, test_datum  = testimageload()
                #test_in_dict = {loss_ph[key]: test_datum[key] for key in loss_ph}
                #test_in_dict[self.image] = test_dataset
                #test_in_dict[self.train_phase] = True                
                #_, test_summary = self.sess.run([self.loss, self.loss_summary], feed_dict=test_in_dict)
                #test_writer.add_summary(test_summary, global_step=i)
            if i % 1000 == 0 and i != 0:
                self.saver.save(self.sess, 'train_weight/'+'Record_2E', global_step=self.global_step)
            if i % 10000 == 0 and i != 0:
                
                eval_tf = EVA_TF(self.sess, self.image, self.train_phase, self.yolopred)
                f = open("mAP.txt", 'a+')
                print>>f, 'Mean AP = {:.4f} i = {}'.format(np.mean(eval_tf.mAP), i)
                f.close()
                print "hahahahah!"
            batch_index += 1   

        train_writer.close()
        return None

    def testAP(self): #TODO add training function!
        EVA_TF(self.sess, self.image, self.train_phase,  self.yolopred)
        return None

def main(argvs):
    yolo = YOLO_TF(argvs)
    cv2.waitKey(1000)


if __name__=='__main__':    
    main(sys.argv)
