import numpy as np
import warnings

""" YOLO regions utilities """

def logistic_activate(x):
  return 1.0/(1.0 + np.exp(-x))

def logistic_gradient(x):
  return (1-x)*x

def yolo_activate_regions(x,num_boxes,num_classes):
    num_coords = 4
    size = num_coords + num_classes + 1
    batch_size,num_filters,height,width = x.shape

    for b in range(0,batch_size):
      for k in range(0,num_boxes):
        # logistic_activate of objectness score
        index = size*k
        x[b, index + 4, :, :] = logistic_activate(x[b, index + 4, :, :])
        # softmax of class prediction
        e_x = np.exp(x[b, index + 5:index + 5 + num_classes, :, :] -
                     np.max(x[b, index + 5:index + 5 + num_classes, :, :],axis=0))
        x[b, index + 5:index + 5 + num_classes, :, :] = e_x/np.sum((e_x),axis=0)
    return x

def yolo_get_region_box(pred,n,i,j,w,h,priors):
  b = np.zeros(4)
  b[0] = (i + logistic_activate(pred[0])) / w
  b[1] = (j + logistic_activate(pred[1])) / h
  b[2] = np.exp(pred[2]) * priors[n,0] / w
  b[3] = np.exp(pred[3]) * priors[n,1] / h
  return b

def yolo_get_region_boxes(x,priors,num_classes,thresh):
    num_boxes = len(priors)
    num_coords = 4
    batch_size,num_filters,height,width = x.shape
    boxes = np.zeros((batch_size,num_boxes*height*width,num_coords))
    probs = np.zeros((batch_size,num_boxes*height*width,num_classes))

    for b in range(0,batch_size):
      for k in range(0,num_boxes):
        for row in range(0,height):
          for col in range(0,width):
            box_idx    = k*width*height+(row*width+col)
            coords_idx = k*(num_coords+num_classes+1)
            # activate of coords, and add prior (biases) w,h
            boxes[b,box_idx,0] = (col + logistic_activate(x[b,coords_idx+0,row,col])) / width
            boxes[b,box_idx,1] = (row + logistic_activate(x[b,coords_idx+1,row,col])) / height
            boxes[b,box_idx,2] = np.exp(x[b,coords_idx+2,row,col]) * priors[k][0] / width
            boxes[b,box_idx,3] = np.exp(x[b,coords_idx+3,row,col]) * priors[k][1] / height
            # scale class probs by bbox objectness score
            scale = x[b,coords_idx+4,row,col]
            probs[b,box_idx,:] = scale*x[b,coords_idx+5:coords_idx+5+num_classes,row,col]
            # set to zero probs under threshold
            probs[b,box_idx,:][probs[b,box_idx,:] < thresh] = 0

    return boxes,probs

def yolo_delta_region_box(delta,truth,pred,n,idx,i,j,w,h,priors,coord_scale):
  p = yolo_get_region_box(pred,n,i,j,w,h,priors)
  iou = yolo_box_iou(p,truth)

  tx = (truth[0]*w - i)
  ty = (truth[1]*h - j)
  tw = np.log(truth[2]*w / priors[n,0])
  th = np.log(truth[3]*h / priors[n,1])

  delta[idx[0],idx[1]+0,idx[2],idx[3]] = coord_scale * (tx - logistic_activate(pred[0])) * logistic_gradient(logistic_activate(pred[0]))
  delta[idx[0],idx[1]+1,idx[2],idx[3]] = coord_scale * (ty - logistic_activate(pred[1])) * logistic_gradient(logistic_activate(pred[1]))
  delta[idx[0],idx[1]+2,idx[2],idx[3]] = coord_scale * (tw - pred[2])
  delta[idx[0],idx[1]+3,idx[2],idx[3]] = coord_scale * (th - pred[3])
  return iou,delta

def yolo_do_nms_sort(boxes,probs,num_classes,nms_thresh):
    batch_size = boxes.shape[0]
    total = boxes.shape[1] # 5*13*13 = 845

    for b in range(0,batch_size):
      for c in range(0,num_classes):
        idx_sort = np.argsort(probs[b,:,c])[::-1]
        for idx_a in range(0,total):
          if probs[b,idx_sort[idx_a],c] == 0: continue
          box_a = boxes[b,idx_sort[idx_a],:]
          for idx_b in range(idx_a+1,total):
            box_b = boxes[b,idx_sort[idx_b],:]
            if (yolo_box_iou(box_a, box_b) > nms_thresh):
              probs[b,idx_sort[idx_b],c] = 0

    return boxes,probs

def yolo_overlap(x1,w1,x2,w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    if(l1 > l2):
        left = l1
    else:
        left = l2
    r1 = x1 + w1/2
    r2 = x2 + w2/2
    if(r1 < r2):
        right = r1
    else:
        right = r2
    return right - left

def yolo_box_intersection(a, b):
    w = yolo_overlap(a[0], a[2], b[0], b[2])
    h = yolo_overlap(a[1], a[3], b[1], b[3])
    if(w < 0 or h < 0):
         return 0
    area = w*h
    return area

def yolo_box_union(a, b):
    i = yolo_box_intersection(a, b)
    u = a[2]*a[3] + b[2]*b[3] - i
    return u

def yolo_box_iou(a, b):
    return yolo_box_intersection(a, b)/yolo_box_union(a, b)


def yolo_draw_detections(impath,boxes,probs,thresh,labels):

    def get_color(c,x,max):
      colors = ( (1,0,1), (0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,0,0) )
      ratio = (float(x)/max)*5
      i = np.floor(ratio)
      j = np.ceil(ratio)
      ratio -= i
      r = (1-ratio) * colors[int(i)][int(c)] + ratio*colors[int(j)][int(c)]
      return r*255

    num_boxes   = boxes.shape[0]
    num_classes = probs.shape[1]

    im  = cv2.imread(impath)

    for i in range(num_boxes):
        #for each box, find the class with maximum prob
        max_class = np.argmax(probs[i,:])
        prob = probs[i,max_class]
        if(prob > thresh):
            print labels[max_class],": ",prob
            b = boxes[i,:]

            left  = (b[0]-b[2]/2.)*im.shape[1]
            right = (b[0]+b[2]/2.)*im.shape[1]
            top   = (b[1]-b[3]/2.)*im.shape[0]
            bot   = (b[1]+b[3]/2.)*im.shape[0]

            if(left < 0): left = 0
            if(right > im.shape[1]-1): right = im.shape[1]-1
            if(top < 0): top = 0
            if(bot > im.shape[0]-1): bot = im.shape[0]-1

            offset = max_class*123457 % len(labels)
            color = (get_color(2,offset,len(labels)),get_color(1,offset,len(labels)),get_color(0,offset,len(labels)))
            cv2.rectangle(im, (int(left),int(top)), (int(right),int(bot)), color, 4)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.65
            thickness = 1
            size=cv2.getTextSize(labels[max_class], font, scale, thickness)
            cv2.rectangle(im, (int(left)-4,int(top)-size[0][1]-8), (int(left)+size[0][0]+8,int(top)), color, -1)
            cv2.putText(im, labels[max_class], (int(left),int(top)-4), font, scale, (0,0,0), thickness, cv2.LINE_AA)

    cv2.imwrite('prediction.jpg',im)
    cv2.imshow('image',im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def yolo_build_gt_batch(batch_gt,image_shape):

    batch_size = len(batch_gt)
    batch_y = np.zeros((batch_size, 5, image_shape[1]/32, image_shape[2]/32))
    batch_y[:,0,:,:] = -1 # indicates nothing on this position

    h = image_shape[1]/32
    w = image_shape[2]/32
    max_truth_boxes = w * h

    for i,gt in enumerate(batch_gt):
        t_ind = 0
        for t in range(min(gt.shape[0],max_truth_boxes)):
            t_i = t_ind%w
            t_j = t_ind/w
            batch_y[i,0,t_j,t_i] = gt[t,0] # object class
            batch_y[i,1,t_j,t_i] = gt[t,1] # x coordinate
            batch_y[i,2,t_j,t_i] = gt[t,2] # y coordinate
            batch_y[i,3,t_j,t_i] = gt[t,3] # width
            batch_y[i,4,t_j,t_i] = gt[t,4] # height
            t_ind += 1

    return batch_y


""" Uitlities to convert Darknet models' weights into keras hdf5 format """

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

#if __name__ == '__main__':
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
