from keras import backend as K
dim_ordering = K.image_dim_ordering()
if dim_ordering == 'th':
    import theano
    from theano import tensor as T
else:
    import tensorflow as tf
    from tensorflow.python.framework import ops

import numpy as np # YOLOLoss is implemented as a numpy function


def cce_flatt(void_class, weights_class):
    def categorical_crossentropy_flatt(y_true, y_pred):
        '''Expects a binary class matrix instead of a vector of scalar classes.
        '''
        if dim_ordering == 'th':
            y_pred = K.permute_dimensions(y_pred, (0, 2, 3, 1))
        shp_y_pred = K.shape(y_pred)
        y_pred = K.reshape(y_pred, (shp_y_pred[0]*shp_y_pred[1]*shp_y_pred[2],
                           shp_y_pred[3]))  # go back to b01,c
        # shp_y_true = K.shape(y_true)

        if dim_ordering == 'th':
            y_true = K.cast(K.flatten(y_true), 'int32')  # b,01 -> b01
        else:
            y_true = K.cast(K.flatten(y_true), 'int32')  # b,01 -> b01

        # remove void classes from cross_entropy
        if len(void_class):
            for i in range(len(void_class)):
                # get idx of non void classes and remove void classes
                # from y_true and y_pred
                idxs = K.not_equal(y_true, void_class[i])
                if dim_ordering == 'th':
                    idxs = idxs.nonzero()
                    y_pred = y_pred[idxs]
                    y_true = y_true[idxs]
                else:
                    y_pred = tf.boolean_mask(y_pred, idxs)
                    y_true = tf.boolean_mask(y_true, idxs)

        if dim_ordering == 'th':
            y_true = T.extra_ops.to_one_hot(y_true, nb_class=y_pred.shape[-1])
        else:
            y_true = tf.one_hot(y_true, K.shape(y_pred)[-1], on_value=1, off_value=0, axis=None, dtype=None, name=None)
            y_true = K.cast(y_true, 'float32')  # b,01 -> b01
        out = K.categorical_crossentropy(y_pred, y_true)

        # Class balancing
        if weights_class is not None:
            weights_class_var = K.variable(value=weights_class)
            class_balance_w = weights_class_var[y_true].astype(K.floatx())
            out = out * class_balance_w

        return K.mean(out)  # b01 -> b,01
    return categorical_crossentropy_flatt


def IoU(n_classes, void_labels):
    def IoU_flatt(y_true, y_pred):
        '''Expects a binary class matrix instead of a vector of scalar classes.
        '''
        if dim_ordering == 'th':
            y_pred = K.permute_dimensions(y_pred, (0, 2, 3, 1))
        shp_y_pred = K.shape(y_pred)
        y_pred = K.reshape(y_pred, (shp_y_pred[0]*shp_y_pred[1]*shp_y_pred[2],
                           shp_y_pred[3]))  # go back to b01,c
        # shp_y_true = K.shape(y_true)
        y_true = K.cast(K.flatten(y_true), 'int32')  # b,01 -> b01
        y_pred = K.argmax(y_pred, axis=-1)

        # We use not_void in case the prediction falls in the void class of
        # the groundtruth
        for i in range(len(void_labels)):
            if i == 0:
                not_void = K.not_equal(y_true, void_labels[i])
            else:
                not_void = not_void * K.not_equal(y_true, void_labels[i])

        sum_I = K.zeros((1,), dtype='float32')

        out = {}
        for i in range(n_classes):
            y_true_i = K.equal(y_true, i)
            y_pred_i = K.equal(y_pred, i)

            if dim_ordering == 'th':
                I_i = K.sum(y_true_i * y_pred_i)
                U_i = K.sum(T.or_(y_true_i, y_pred_i) * not_void)
                # I = T.set_subtensor(I[i], I_i)
                # U = T.set_subtensor(U[i], U_i)
                sum_I = sum_I + I_i
            else:
                U_i = K.sum(K.cast(tf.logical_and(tf.logical_or(y_true_i, y_pred_i), not_void), 'float32'))
                y_true_i = K.cast(y_true_i, 'float32')
                y_pred_i = K.cast(y_pred_i, 'float32')
                I_i = K.sum(y_true_i * y_pred_i)
                sum_I = sum_I + I_i
            out['I'+str(i)] = I_i
            out['U'+str(i)] = U_i

        if dim_ordering == 'th':
            accuracy = K.sum(sum_I) / K.sum(not_void)
        else:
            accuracy = K.sum(sum_I) / tf.reduce_sum(tf.cast(not_void, 'float32'))
        out['acc'] = accuracy
        return out
    return IoU_flatt

"""YOLO loss function"""

from tools.yolo_utils import logistic_activate,logistic_gradient,yolo_activate_regions
from tools.yolo_utils import yolo_delta_region_box,yolo_box_iou,yolo_get_region_box 
from tools.yolo_utils import yolo_get_region_boxes,yolo_do_nms_sort

def YOLOLoss(input_shape=(3,640,640),num_classes=45,priors=[[0.25,0.25], [0.5,0.5], [1.0,1.0], [1.7,1.7], [2.5,2.5]],max_truth_boxes=30,thresh=0.6,object_scale=5.0,noobject_scale=1.0,coord_scale=1.0,class_scale=1.0):

  # Def custom loss function using numpy
  def _YOLOLoss(y_true, y_pred, name=None):

    with ops.name_scope( name, "YOLOloss", [y_true,y_pred] ) as name:
        delta = py_func(YOLOLoss_np,
                        [y_true,y_pred,num_classes,np.array(priors),
                         max_truth_boxes,thresh,object_scale,
                         noobject_scale,coord_scale,class_scale],
                        [tf.float32],
                        name=name,
                        grad=_YOLOLossGrad)  # <-- call to the gradient

        output_shape = [-1, len(priors)*(4+num_classes+1), input_shape[1]/32, input_shape[2]/32]
        delta = tf.reshape(delta[0], output_shape)
        return K.sum(K.square(delta),axis=(1,2,3))

  return _YOLOLoss

# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _YOLOLossGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

# Actual gradient:
def _YOLOLossGrad(op, grad):
    y_true = op.inputs[0]
    y_pred = op.inputs[1]
    num_classes = op.inputs[2]
    priors = op.inputs[3]
    max_truth_boxes = op.inputs[4]
    thresh = op.inputs[5]
    object_scale = op.inputs[6]
    noobject_scale = op.inputs[7]
    coord_scale = op.inputs[8]
    class_scale = op.inputs[9]
    return [tf.zeros_like(y_true), -op.outputs[0], tf.zeros_like(num_classes), tf.zeros_like(priors),
            tf.zeros_like(max_truth_boxes),tf.zeros_like(thresh),tf.zeros_like(object_scale),
            tf.zeros_like(noobject_scale),tf.zeros_like(coord_scale),tf.zeros_like(class_scale)]


def YOLOLoss_np(y_true, y_pred, num_classes, priors, max_truth_boxes, thresh,
                object_scale, noobject_scale, coord_scale, class_scale):

    num_priors = priors.shape[0]
    prior_match = True
    num_coords  = 4
    data_size = num_coords+num_classes+1
    delta_shape = y_pred.shape
    delta = np.zeros(delta_shape, dtype='f')

    y_pred = yolo_activate_regions(y_pred,num_priors,num_classes)

    batch_size = y_pred.shape[0]
    h = y_pred.shape[2]
    w = y_pred.shape[3]

    avg_iou = 0.
    recall = 0.
    avg_cat = 0.
    avg_obj = 0.
    avg_anyobj = 0.
    count = 0.
    class_count = 0.

    for b in range(batch_size):
      for j in range(h):
        for i in range(w):
          for n in range(num_priors):
            pred_box = yolo_get_region_box(y_pred[b,n*data_size:n*data_size+4,j,i],n,i,j,w,h,priors)
            best_iou = 0.0
            for t in range(max_truth_boxes):
              t_i = t%w
              t_j = t/w
              if (y_true[b,0,t_j,t_i] < 0): break # no truth information in this position
              truth_box = y_true[b,1:5,t_j,t_i]
              iou = yolo_box_iou(pred_box, truth_box)
              if (iou > best_iou):
                best_iou = iou
            avg_anyobj += y_pred[b,n*data_size+4,j,i]
            if (best_iou > thresh):
              delta[b,n*data_size+4,j,i] = 0
            else:
              delta[b,n*data_size+4,j,i] = noobject_scale * ((0 - y_pred[b,n*data_size+4,j,i]) * logistic_gradient(y_pred[b,n*data_size+4,j,i]))

      for t in range(max_truth_boxes):
        t_i = t%w
        t_j = t/w
        if (y_true[b,0,t_j,t_i] < 0): break # no truth information in this position
        truth_box = y_true[b,1:5,t_j,t_i]

        best_iou = 0
        best_index = (0,0,0,0)
        best_n = 0
        i = int(truth_box[0] * w)
        j = int(truth_box[1] * h)
        truth_shift = truth_box.copy()
        truth_shift[0] = 0
        truth_shift[1] = 0
        for n in range(num_priors):
          pred_box = yolo_get_region_box(y_pred[b,n*data_size:n*data_size+4,j,i],n,i,j,w,h,priors)
          if (prior_match):
            pred_box[2] = priors[n,0]/w
            pred_box[3] = priors[n,1]/h
          pred_box[0] = 0
          pred_box[1] = 0
          iou = yolo_box_iou(pred_box, truth_shift)
          if (iou > best_iou):
            best_index = (b,n*data_size,j,i)
            best_iou = iou
            best_n = n

        iou,delta = yolo_delta_region_box(delta, truth_box, y_pred[b,best_n*data_size:best_n*data_size+4,j,i], best_n, best_index, i, j, w, h, priors, coord_scale)

        if(iou > .5): recall += 1.0
        avg_iou += iou
        avg_obj += y_pred[best_index[0],best_index[1]+4,best_index[2],best_index[3]]

        x = y_pred[best_index[0],best_index[1] + 4,best_index[2],best_index[3]]
        delta[best_index[0],best_index[1] + 4,best_index[2],best_index[3]] = object_scale * (iou - x) * logistic_gradient(x)

        truth_class = y_true[b,0,t_j,t_i]
        #delta_region_class(best_index + 5, truth_class)
        for c in range(num_classes):
          if(c == truth_class):
            delta[best_index[0],best_index[1] + 5 + c,best_index[2],best_index[3]] = class_scale * (1 - y_pred[best_index[0],best_index[1] + 5 + c,best_index[2],best_index[3]])
            avg_cat += y_pred[best_index[0],best_index[1] + 5 + c,best_index[2],best_index[3]]
          else:
            delta[best_index[0],best_index[1] + 5 + c,best_index[2],best_index[3]] = class_scale * (0 - y_pred[best_index[0],best_index[1] + 5 + c,best_index[2],best_index[3]])
        count += 1
        class_count +=1

    #print "Region Avg IOU: ",avg_iou/count,", Class: ",avg_cat/class_count,", Obj: ",avg_obj/count,", No Obj: ",avg_anyobj/(w*h*num_priors*batch_size),", Avg Recall: ",recall/count,",  count: ",count
    #loss = np.power(np.linalg.norm(delta), 2)
    #print "Loss ",loss

    return delta # return the gradient (the actual loss is just reduce_sum(square(delta)))


"""YOLO f-score detection metric"""

def YOLOFscore(input_shape=(3,640,640),num_classes=45,priors=[[0.25,0.25], [0.5,0.5], [1.0,1.0], [1.7,1.7], [2.5,2.5]],max_truth_boxes=30,thresh=0.6,nms_thresh=0.3):

  # Def custom metric using numpy
  def _YOLOFscore(y_true, y_pred, name=None):
    with ops.name_scope( name, "YOLOFscore", [y_true,y_pred] ) as name:
      fscore = tf.py_func(YOLOFscore_np,
                          [y_true,y_pred,num_classes,np.array(priors),
                           max_truth_boxes,thresh,nms_thresh],
                          [tf.float32], name=name)
    return {'fscore':tf.reduce_mean(fscore[0])}

  return _YOLOFscore


def YOLOFscore_np(y_true, y_pred, num_classes, priors, max_truth_boxes, thresh, nms_thresh):
    batch_size = y_pred.shape[0]
    fscore = np.zeros(batch_size, dtype='f')
    num_priors = priors.shape[0]
    num_coords  = 4
    data_size = num_coords+num_classes+1

    y_pred = yolo_activate_regions(y_pred, num_priors, num_classes)
    boxes,probs = yolo_get_region_boxes(y_pred, priors, num_classes, thresh)
    boxes,probs = yolo_do_nms_sort(boxes, probs, num_classes, nms_thresh)


    # for each image in batch
    for i in range(batch_size):
        b = boxes[i,:,:]
        p = probs[i,:,:]
        num_boxes   = b.shape[0]
        num_classes = p.shape[1]
        # put GT boxes for this image in a list
        gt_boxes = []
        for h_i in range(y_true.shape[2]):
          for w_i in range(y_true.shape[3]):
            if y_true[i,0,h_i,w_i] >= 0:
              gt_boxes.append(y_true[i,:,h_i,w_i])
        num_gt = len(gt_boxes)
        ok    = 0.
        total = 0.
        # for each detected bounding box in this image, find the class with maximum prob
        for j in range(num_boxes):
            max_class = np.argmax(p[j,:])
            pb = p[j,max_class]
            if(pb > thresh):
                total += 1.
                bb = b[j,:]
                # count as TP if the box overlaps more than 50% with a GT object and the class is correct
                for idx,gt in enumerate(gt_boxes):
                    if gt[0] == max_class and yolo_box_iou(bb,gt[1:5]) > 0.5:
                        ok += 1.
                        gt_boxes = gt_boxes[0:idx]+gt_boxes[idx+1:]
                        break
        recall = ok / num_gt
        precision = 0.
        if total > 0.:
          precision = ok / total
        if (recall+precision) > 0:
          fscore[i] = 2 * ((recall*precision) / (recall+precision))

    return fscore
