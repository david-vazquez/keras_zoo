import numpy as np
import warnings
import math
import cv2

"""
    YOLO utitlities
    code adapted from https://github.com/thtrieu/darkflow/
                 and  https://pjreddie.com/darknet/yolo/
"""

def yolo_build_gt_batch(batch_gt,image_shape,num_classes,num_priors=5):

    h = image_shape[1]/32
    w = image_shape[2]/32
    c = num_classes
    b = num_priors  # TODO pass num_priors
    batch_size = len(batch_gt)
    batch_y = np.zeros([batch_size,h*w,b,c+4+1+1+2+2])

    cellx = 32
    celly = 32
    for i,gt in enumerate(batch_gt):
        if gt.shape[0] == 0:
          # if there are no objects we'll get NaNs on YOLOLoss, set everything to one!
          # TODO check if the following line harms learning in case of 
          #      having lots of images with no objects
          batch_y[i] = np.ones((h*w,b,c+4+1+1+2+2))
          continue
        objects = gt.tolist()
        for obj in objects:
            centerx = obj[1] * image_shape[2]
            centery = obj[2] * image_shape[1]
            cx = centerx / cellx
            cy = centery / celly
            obj[1] = cx - np.floor(cx) # centerx
            obj[2] = cy - np.floor(cy) # centerx
            obj[3] = np.sqrt(obj[3])
            obj[4] = np.sqrt(obj[4])
            obj += [int(np.floor(cy) * w + np.floor(cx))]

        probs = np.zeros([h*w,b,c])
        confs = np.zeros([h*w,b,1])
        coord = np.zeros([h*w,b,4])
        prear = np.zeros([h*w,4])

        for obj in objects:
            probs[obj[5], :, :] = [[0.]*c] * b
            probs[obj[5], :, int(obj[0])] = 1.
            coord[obj[5], :, :] = [obj[1:5]] * b
            prear[obj[5],0] = obj[1] - obj[3]**2 * .5 * w # xleft
            prear[obj[5],1] = obj[2] - obj[4]**2 * .5 * h # yup
            prear[obj[5],2] = obj[1] + obj[3]**2 * .5 * w # xright
            prear[obj[5],3] = obj[2] + obj[4]**2 * .5 * h # ybot
            confs[obj[5], :, 0] = [1.] * b

        upleft   = np.expand_dims(prear[:,0:2], 1)
        botright = np.expand_dims(prear[:,2:4], 1)
        wh = botright - upleft
        area = wh[:,:,0] * wh[:,:,1]
        upleft   = np.concatenate([upleft] * b, 1)
        botright = np.concatenate([botright] * b, 1)
        areas = np.concatenate([area] * b, 1)

        batch_y[i,:] = np.concatenate((probs,confs,coord,areas[:,:,np.newaxis],upleft,botright),axis=2)

    return batch_y


class BoundBox:
    def __init__(self, classes):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.class_num = classes
        self.probs = np.zeros((classes,))

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);

def prob_compare(box):
    return box.probs[box.class_num]

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def yolo_postprocess_net_out(net_out, anchors, labels, threshold, nms_threshold):
	C = len(labels) 
        B = len(anchors)
        net_out = np.transpose(net_out, (1,2,0))
	H,W = net_out.shape[:2]
	net_out = net_out.reshape([H, W, B, -1])

	boxes = list()
	for row in range(H):
		for col in range(W):
			for b in range(B):
				bx = BoundBox(C)
				bx.x, bx.y, bx.w, bx.h, bx.c = net_out[row, col, b, :5]
				bx.c = expit(bx.c)
				bx.x = (col + expit(bx.x)) / W
				bx.y = (row + expit(bx.y)) / H
				bx.w = math.exp(bx.w) * anchors[b][0] / W
				bx.h = math.exp(bx.h) * anchors[b][1] / H
				classes = net_out[row, col, b, 5:]
				bx.probs = _softmax(classes) * bx.c
				bx.probs *= bx.probs > threshold
				boxes.append(bx)

	# non max suppress boxes
	for c in range(C):
		for i in range(len(boxes)):
			boxes[i].class_num = c
		boxes = sorted(boxes, key = prob_compare)
		for i in range(len(boxes)):
			boxi = boxes[i]
			if boxi.probs[c] == 0: continue
			for j in range(i + 1, len(boxes)):
				boxj = boxes[j]
				if box_iou(boxi, boxj) >= nms_threshold:
					boxes[j].probs[c] = 0.

	return boxes

def yolo_draw_detections(boxes, im, anchors, labels, threshold, nms_threshold):

        def get_color(c,x,max):
          colors = ( (1,0,1), (0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,0,0) )
          ratio = (float(x)/max)*5
          i = np.floor(ratio)
          j = np.ceil(ratio)
          ratio -= i
          r = (1-ratio) * colors[int(i)][int(c)] + ratio*colors[int(j)][int(c)]
          return r*255

	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	for b in boxes:
		max_indx = np.argmax(b.probs)
		max_prob = b.probs[max_indx]
		label = 'object' * int(len(labels) < 2)
		label += labels[max_indx] * int(len(labels)>1)
		if max_prob > threshold:
			left  = int ((b.x - b.w/2.) * w)
			right = int ((b.x + b.w/2.) * w)
			top   = int ((b.y - b.h/2.) * h)
			bot   = int ((b.y + b.h/2.) * h)
			if left  < 0    :  left = 0
			if right > w - 1: right = w - 1
			if top   < 0    :   top = 0
			if bot   > h - 1:   bot = h - 1
			thick = int((h+w)/300)
			mess = '{}'.format(label)
                        offset = max_indx*123457 % len(labels)
                        color = (get_color(2,offset,len(labels)),
                                 get_color(1,offset,len(labels)),
                                 get_color(0,offset,len(labels)))
			cv2.rectangle(imgcv,
				(left, top), (right, bot),
				color, thick)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale = 0.65
                        thickness = 1
                        size=cv2.getTextSize(mess, font, scale, thickness)
                        cv2.rectangle(im, (left-2,top-size[0][1]-4), (left+size[0][0]+4,top), color, -1)
                        cv2.putText(im, mess, (left+2,top-2), font, scale, (0,0,0), thickness, cv2.LINE_AA)
	return imgcv


""" 
   Utilities to convert Darknet models' weights into keras hdf5 format
   code adapted from https://github.com/sunshineatnoon/Darknet.keras
"""

class dummy_layer:
    def __init__(self,size,c,n,h,w,type):
        self.size = size
        self.c = c
        self.n = n
        self.h = h
        self.w = w
        self.type = type

class dummy_convolutional_layer(dummy_layer):
    def __init__(self,size,c,n,h,w):
        dummy_layer.__init__(self,size,c,n,h,w,"CONVOLUTIONAL")
        self.biases = np.zeros(n)
        self.weights = np.zeros((size*size,c,n))

class dummy_connected_layer(dummy_layer):
    def __init__(self,size,c,n,h,w,input_size,output_size):
        dummy_layer.__init__(self,size,c,n,h,w,"CONNECTED")
        self.output_size = output_size
        self.input_size = input_size
        self.biases = np.zeros(output_size)
        self.weights = np.zeros((output_size*input_size))

class dummy_YOLO:
    layer_number = 48
    def __init__(self,num_classes=80,num_priors=5):
        self.layers = []
        self.num_classes = num_classes
        self.num_priors = num_priors
        self.layers.append(dummy_convolutional_layer(3,3,32,416,416))
        self.layers.append(dummy_layer(0,0,32,0,0,"BATCHNORM"))
        self.layers.append(dummy_layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(dummy_convolutional_layer(3,32,64,208,208))
        self.layers.append(dummy_layer(0,0,64,0,0,"BATCHNORM"))
        self.layers.append(dummy_layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(dummy_convolutional_layer(3,64,128,104,104))
        self.layers.append(dummy_layer(0,0,128,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(1,128,64,104,104))
        self.layers.append(dummy_layer(0,0,64,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(3,64,128,104,104))
        self.layers.append(dummy_layer(0,0,128,0,0,"BATCHNORM"))
        self.layers.append(dummy_layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(dummy_convolutional_layer(3,128,256,52,52))
        self.layers.append(dummy_layer(0,0,256,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(1,256,128,52,52))
        self.layers.append(dummy_layer(0,0,128,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(3,128,256,52,52))
        self.layers.append(dummy_layer(0,0,256,0,0,"BATCHNORM"))
        self.layers.append(dummy_layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(dummy_convolutional_layer(3,256,512,26,26))
        self.layers.append(dummy_layer(0,0,512,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(1,512,256,26,26))
        self.layers.append(dummy_layer(0,0,256,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(3,256,512,26,26))
        self.layers.append(dummy_layer(0,0,512,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(1,512,256,26,26))
        self.layers.append(dummy_layer(0,0,256,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(3,256,512,26,26))
        self.layers.append(dummy_layer(0,0,512,0,0,"BATCHNORM"))
        self.layers.append(dummy_layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(dummy_convolutional_layer(3,512,1024,13,13))
        self.layers.append(dummy_layer(0,0,1024,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(1,1024,512,13,13))
        self.layers.append(dummy_layer(0,0,512,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(3,512,1024,13,13))
        self.layers.append(dummy_layer(0,0,1024,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(1,1024,512,13,13))
        self.layers.append(dummy_layer(0,0,512,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(3,512,1024,13,13))
        self.layers.append(dummy_layer(0,0,1024,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(3,1024,1024,13,13))
        self.layers.append(dummy_layer(0,0,1024,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(3,1024,1024,13,13))
        self.layers.append(dummy_layer(0,0,1024,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(3,3072,1024,13,13))
        self.layers.append(dummy_layer(0,0,1024,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(1,1024,self.num_priors*(4+self.num_classes+1),13,13))

class dummy_TinyYOLO:
    layer_number = 23
    def __init__(self,num_classes=80,num_priors=5):
        self.layers = []
        self.num_classes = num_classes
        self.num_priors = num_priors
        self.layers.append(dummy_convolutional_layer(3,3,16,416,416))
        self.layers.append(dummy_layer(0,0,16,0,0,"BATCHNORM"))
        self.layers.append(dummy_layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(dummy_convolutional_layer(3,16,32,208,208))
        self.layers.append(dummy_layer(0,0,32,0,0,"BATCHNORM"))
        self.layers.append(dummy_layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(dummy_convolutional_layer(3,32,64,104,104))
        self.layers.append(dummy_layer(0,0,64,0,0,"BATCHNORM"))
        self.layers.append(dummy_layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(dummy_convolutional_layer(3,64,128,52,52))
        self.layers.append(dummy_layer(0,0,128,0,0,"BATCHNORM"))
        self.layers.append(dummy_layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(dummy_convolutional_layer(3,128,256,26,26))
        self.layers.append(dummy_layer(0,0,256,0,0,"BATCHNORM"))
        self.layers.append(dummy_layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(dummy_convolutional_layer(3,256,512,13,13))
        self.layers.append(dummy_layer(0,0,512,0,0,"BATCHNORM"))
        self.layers.append(dummy_layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(dummy_convolutional_layer(3,512,1024,13,13))
        self.layers.append(dummy_layer(0,0,1024,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(3,1024,1024,13,13))
        self.layers.append(dummy_layer(0,0,1024,0,0,"BATCHNORM"))
        self.layers.append(dummy_convolutional_layer(1,1024,self.num_priors*(4+self.num_classes+1),13,13))

def ReadYOLONetWeights(d_model,weight_path):

    type_string = "(3)float32,i4,"

    for i in range(d_model.layer_number):
        l = d_model.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            bias_number = l.n
            weight_number = l.n*l.c*l.size*l.size
            type_string = type_string + "("+ str(bias_number) + ")float32,"
            if(i != d_model.layer_number-1):
                scales_number    = l.n
                means_number     = l.n
                variances_number = l.n
                type_string = type_string + ("("+ str(scales_number) +
                              ")float32,(" + str(means_number) +
                              ")float32,(" + str(variances_number) + ")float32,")
            type_string = type_string +"(" + str(weight_number) + ")float32"
            if(i != d_model.layer_number-1):
                type_string = type_string + ","

    dt = np.dtype(type_string)
    testArray = np.fromfile(weight_path,dtype=dt)

    count = 2
    for i in range(0,d_model.layer_number):
        l = d_model.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            l.biases = np.asarray(testArray[0][count])
            count = count + 1
            if(i != d_model.layer_number-1): #if not last layer read batch normalization weights
                bn = d_model.layers[i+1]
                bn.weights = [np.asarray(testArray[0][count]),
                              np.zeros((np.asarray(testArray[0][count]).shape),
                              dtype=np.float32),np.asarray(testArray[0][count+1]),
                              np.asarray(testArray[0][count+2])]
                count = count + 3
            l.weights = np.asarray(testArray[0][count])
            count = count + 1
            d_model.layers[i] = l

    #write back to file and see if it is the same
    '''
    write_fp = open('reconstruct.weights','w')
    write_fp.write((np.asarray(testArray[0][0])).tobytes())
    write_fp.write((np.asarray(testArray[0][1])).tobytes())
    for i in range(0,d_model.layer_number):
        l = d_model.layers[i]
        if(l.type == "CONVOLUTIONAL" or l.type == "CONNECTED"):
            write_fp.write(l.biases.tobytes())
            if(i != d_model.layer_number-1):
              write_fp.write(d_model.layers[i+1].weights[0].tobytes())
              write_fp.write(d_model.layers[i+1].weights[2].tobytes())
              write_fp.write(d_model.layers[i+1].weights[3].tobytes())
            write_fp.write(l.weights.tobytes())
    write_fp.close()
    '''
    #reshape weights in every layer
    for i in range(d_model.layer_number):
        l = d_model.layers[i]
        if(l.type == 'CONVOLUTIONAL'):
            weight_array = l.weights
            n = weight_array.shape[0]
            weight_array = np.reshape(weight_array,[l.n,l.c,l.size,l.size])
            l.weights = weight_array

    return d_model

def DarknetToKerasYOLO(yoloNet):

    K.set_image_dim_ordering('th')

    net={}
    input_tensor = Input(shape=(3,416,416))
    net['input'] = input_tensor
    l = yoloNet.layers[0]
    lbn = yoloNet.layers[1]
    net['conv1'] = (YOLOConvolution2D(32, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['input'])
    net['relu1'] = (LeakyReLU(alpha=0.1))(net['conv1'])
    net['pool1'] = (MaxPooling2D(pool_size=(2, 2),border_mode='valid'))(net['relu1'])
    l = yoloNet.layers[3]
    lbn = yoloNet.layers[4]
    net['conv2'] = (YOLOConvolution2D(64, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['pool1'])
    net['relu2'] = (LeakyReLU(alpha=0.1))(net['conv2'])
    net['pool2'] = (MaxPooling2D(pool_size=(2, 2),border_mode='valid'))(net['relu2'])
    l = yoloNet.layers[6]
    lbn = yoloNet.layers[7]
    net['conv3_1'] = (YOLOConvolution2D(128, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['pool2'])
    net['relu3_1'] = (LeakyReLU(alpha=0.1))(net['conv3_1'])
    l = yoloNet.layers[8]
    lbn = yoloNet.layers[9]
    net['conv3_2'] = (YOLOConvolution2D(64, 1, 1, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['relu3_1'])
    net['relu3_2'] = (LeakyReLU(alpha=0.1))(net['conv3_2'])
    l = yoloNet.layers[10]
    lbn = yoloNet.layers[11]
    net['conv3_3'] = (YOLOConvolution2D(128, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['relu3_2'])
    net['relu3_3'] = (LeakyReLU(alpha=0.1))(net['conv3_3'])
    net['pool3'] = (MaxPooling2D(pool_size=(2, 2),border_mode='valid'))(net['relu3_3'])
    l = yoloNet.layers[13]
    lbn = yoloNet.layers[14]
    net['conv4_1'] = (YOLOConvolution2D(256, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['pool3'])
    net['relu4_1'] = (LeakyReLU(alpha=0.1))(net['conv4_1'])
    l = yoloNet.layers[15]
    lbn = yoloNet.layers[16]
    net['conv4_2'] = (YOLOConvolution2D(128, 1, 1, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['relu4_1'])
    net['relu4_2'] = (LeakyReLU(alpha=0.1))(net['conv4_2'])
    l = yoloNet.layers[17]
    lbn = yoloNet.layers[18]
    net['conv4_3'] = (YOLOConvolution2D(256, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['relu4_2'])
    net['relu4_3'] = (LeakyReLU(alpha=0.1))(net['conv4_3'])
    net['pool4'] = (MaxPooling2D(pool_size=(2, 2),border_mode='valid'))(net['relu4_3'])
    l = yoloNet.layers[20]
    lbn = yoloNet.layers[21]
    net['conv5_1'] = (YOLOConvolution2D(512, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['pool4'])
    net['relu5_1'] = (LeakyReLU(alpha=0.1))(net['conv5_1'])
    l = yoloNet.layers[22]
    lbn = yoloNet.layers[23]
    net['conv5_2'] = (YOLOConvolution2D(256, 1, 1, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['relu5_1'])
    net['relu5_2'] = (LeakyReLU(alpha=0.1))(net['conv5_2'])
    l = yoloNet.layers[24]
    lbn = yoloNet.layers[25]
    net['conv5_3'] = (YOLOConvolution2D(512, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['relu5_2'])
    net['relu5_3'] = (LeakyReLU(alpha=0.1))(net['conv5_3'])
    l = yoloNet.layers[26]
    lbn = yoloNet.layers[27]
    net['conv5_4'] = (YOLOConvolution2D(256, 1, 1, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['relu5_3'])
    net['relu5_4'] = (LeakyReLU(alpha=0.1))(net['conv5_4'])
    l = yoloNet.layers[28]
    lbn = yoloNet.layers[29]
    net['conv5_5'] = (YOLOConvolution2D(512, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['relu5_4'])
    net['relu5_5'] = (LeakyReLU(alpha=0.1))(net['conv5_5'])
    net['pool5'] = (MaxPooling2D(pool_size=(2, 2),border_mode='valid'))(net['relu5_5'])
    l = yoloNet.layers[31]
    lbn = yoloNet.layers[32]
    net['conv6_1'] = (YOLOConvolution2D(1024, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['pool5'])
    net['relu6_1'] = (LeakyReLU(alpha=0.1))(net['conv6_1'])
    l = yoloNet.layers[33]
    lbn = yoloNet.layers[34]
    net['conv6_2'] = (YOLOConvolution2D(512, 1, 1, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['relu6_1'])
    net['relu6_2'] = (LeakyReLU(alpha=0.1))(net['conv6_2'])
    l = yoloNet.layers[35]
    lbn = yoloNet.layers[36]
    net['conv6_3'] = (YOLOConvolution2D(1024, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['relu6_2'])
    net['relu6_3'] = (LeakyReLU(alpha=0.1))(net['conv6_3'])
    l = yoloNet.layers[37]
    lbn = yoloNet.layers[38]
    net['conv6_4'] = (YOLOConvolution2D(512, 1, 1, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['relu6_3'])
    net['relu6_4'] = (LeakyReLU(alpha=0.1))(net['conv6_4'])
    l = yoloNet.layers[39]
    lbn = yoloNet.layers[40]
    net['conv6_5'] = (YOLOConvolution2D(1024, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['relu6_4'])
    net['relu6_5'] = (LeakyReLU(alpha=0.1))(net['conv6_5'])
    l = yoloNet.layers[41]
    lbn = yoloNet.layers[42]
    net['conv6_6'] = (YOLOConvolution2D(1024, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['relu6_5'])
    net['relu6_6'] = (LeakyReLU(alpha=0.1))(net['conv6_6'])
    l = yoloNet.layers[43]
    lbn = yoloNet.layers[44]
    net['conv6_7'] = (YOLOConvolution2D(1024, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['relu6_6'])
    net['relu6_7'] = (LeakyReLU(alpha=0.1))(net['conv6_7'])
    net['reshape7'] = (Reshape((2048,13,13)))(net['relu5_5'])
    net['merge7'] = (merge([net['reshape7'], net['relu6_7']], mode='concat', concat_axis=1))
    l = yoloNet.layers[45]
    lbn = yoloNet.layers[46]
    net['conv8'] = (YOLOConvolution2D(1024, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001, weights=[l.weights,l.biases]+lbn.weights))(net['merge7'])
    net['relu8'] = (LeakyReLU(alpha=0.1))(net['conv8'])
    l = yoloNet.layers[47]
    net['conv9'] = (Convolution2D(yoloNet.num_priors*(4+yoloNet.num_classes+1), 1, 1, border_mode='same',subsample=(1,1),weights=[l.weights,l.biases]))(net['relu8'])

    model = Model(net['input'], net['conv9'])
    return model

def DarknetToKerasTinyYOLO(yoloNet):
    model = Sequential()

    K.set_image_dim_ordering('th')

    #Use a for loop to replace all manually defined layers
    for i in range(0,yoloNet.layer_number):
        l = yoloNet.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            if i<yoloNet.layer_number-1: # all conv layers but the last do batch_normalization
              lbn = yoloNet.layers[i+1]
              if i==0: # input layer must define an input_shape
                model.add(YOLOConvolution2D(l.n, l.size, l.size,
                                            weights=[l.weights,l.biases]+lbn.weights,
                                            border_mode='same',subsample=(1,1),
                                            input_shape=(3,416,416),epsilon=0.000001))
              else:
                model.add(YOLOConvolution2D(l.n, l.size, l.size,
                                            weights=[l.weights,l.biases]+lbn.weights,
                                            border_mode='same',subsample=(1,1),
                                            epsilon=0.000001))
              model.add(LeakyReLU(alpha=0.1))
            else:
              model.add(Convolution2D(l.n, l.size, l.size, weights=[l.weights,l.biases],
                                      border_mode='same',subsample=(1,1)))
              model.add(Activation('linear'))

        elif(l.type == "MAXPOOL"):
            if (i==17) : #17th layer in tinyYOLO has adifferent stride
              model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1),border_mode='same'))
            else:
              model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        else:
            pass
    return model

#Example use of Darknet to Keras converter
#
#    dummy_model = dummy_TinyYOLO()
#    dummy_model = ReadYOLONetWeights(dummy_model,'weights/tiny-yolo.weights')
#    model = DarknetToKerasTinyYOLO(dummy_model)
#    model.save_weights('weights/tiny-yolo.hdf5')
#    print "weights/tiny-yolo.weights converted to weights/tiny-yolo.hdf5"
#
#    dummy_model = dummy_YOLO()
#    dummy_model = ReadYOLONetWeights(dummy_model,'weights/yolo.weights')
#    model = DarknetToKerasYOLO(dummy_model)
#    model.save_weights('weights/yolo.hdf5')
#    print "weights/yolo.weights converted to weights/yolo.hdf5"
#
#    dummy_model_voc = dummy_YOLO(num_classes=20)
#    dummy_model_voc = ReadYOLONetWeights(dummy_model_voc,'weights/yolo-voc.weights')
#    model_voc = DarknetToKerasYOLO(dummy_model_voc)
#    model_voc.save_weights('weights/yolo-voc.hdf5')
#    print "weights/yolo-voc.weights converted to weights/yolo-voc.hdf5"
#
#    dummy_model_tt100k = dummy_YOLO(num_classes=45)
#    dummy_model_tt100k = ReadYOLONetWeights(dummy_model_tt100k,'weights/yolo-tt100k_45000.weights')
#    model_tt100k = DarknetToKerasYOLO(dummy_model_tt100k)
#    model_tt100k.save_weights('weights/yolo-tt100k_45000.hdf5')
#    print "weights/yolo-tt100k.weights converted to weights/yolo-tt100k.hdf5"
