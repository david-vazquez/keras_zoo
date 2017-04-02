import numpy as np
from keras import backend as K
dim_ordering = K.image_dim_ordering()
if dim_ordering == 'th':
    import theano
    from theano import tensor as T
else:
    import tensorflow as tf
    from tensorflow.python.framework import ops

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



"""
    YOLO loss function
    code adapted from https://github.com/thtrieu/darkflow/
"""

def logistic_activate_tensor(x):
    return 1. / (1. + tf.exp(-x))

def YOLOLoss(input_shape=(3,640,640),num_classes=45,priors=[[0.25,0.25], [0.5,0.5], [1.0,1.0], [1.7,1.7], [2.5,2.5]],max_truth_boxes=30,thresh=0.6,object_scale=5.0,noobject_scale=1.0,coord_scale=1.0,class_scale=1.0):

  # Def custom loss function using numpy
  def _YOLOLoss(y_true, y_pred, name=None, priors=priors):

      net_out = tf.transpose(y_pred, perm=[0, 2, 3, 1])

      _,h,w,c = net_out.get_shape().as_list()
      b = len(priors)
      anchors = np.array(priors)

      _probs, _confs, _coord, _areas, _upleft, _botright = tf.split(y_true, [num_classes,1,4,1,2,2], axis=3)
      _confs = tf.squeeze(_confs,3)
      _areas = tf.squeeze(_areas,3)

      net_out_reshape = tf.reshape(net_out, [-1, h, w, b, (4 + 1 + num_classes)])
      # Extract the coordinate prediction from net.out
      coords = net_out_reshape[:, :, :, :, :4]
      coords = tf.reshape(coords, [-1, h*w, b, 4])
      adjusted_coords_xy = logistic_activate_tensor(coords[:,:,:,0:2])
      adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, b, 2]) / np.reshape([w, h], [1, 1, 1, 2]))
      coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

      adjusted_c = logistic_activate_tensor(net_out_reshape[:, :, :, :, 4])
      adjusted_c = tf.reshape(adjusted_c, [-1, h*w, b, 1])

      adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
      adjusted_prob = tf.reshape(adjusted_prob, [-1, h*w, b, num_classes])

      adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

      wh = tf.pow(coords[:,:,:,2:4], 2) *  np.reshape([w, h], [1, 1, 1, 2])
      area_pred = wh[:,:,:,0] * wh[:,:,:,1]
      centers = coords[:,:,:,0:2]
      floor = centers - (wh * .5)
      ceil  = centers + (wh * .5)

      # calculate the intersection areas
      intersect_upleft   = tf.maximum(floor, _upleft)
      intersect_botright = tf.minimum(ceil, _botright)
      intersect_wh = intersect_botright - intersect_upleft
      intersect_wh = tf.maximum(intersect_wh, 0.0)
      intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

      # calculate the best IOU, set 0.0 confidence for worse boxes
      iou = tf.truediv(intersect, _areas + area_pred - intersect)
      best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
      #best_box = tf.grater_than(iou, 0.5) # LLUIS ???
      best_box = tf.to_float(best_box)
      confs = tf.multiply(best_box, _confs)

      # take care of the weight terms
      conid = noobject_scale * (1. - confs) + object_scale * confs
      weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
      cooid = coord_scale * weight_coo
      weight_pro = tf.concat(num_classes * [tf.expand_dims(confs, -1)], 3)
      proid = class_scale * weight_pro

      true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs ], 3)
      wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid ], 3)

      loss = tf.pow(adjusted_net_out - true, 2)
      loss = tf.multiply(loss, wght)
      loss = tf.reshape(loss, [-1, h*w*b*(4 + 1 + num_classes)])
      loss = tf.reduce_sum(loss, 1)

      return .5*tf.reduce_mean(loss)

  return _YOLOLoss


"""
    YOLO detection metrics
    code adapted from https://github.com/thtrieu/darkflow/
"""
def YOLOMetrics(input_shape=(3,640,640),num_classes=45,priors=[[0.25,0.25], [0.5,0.5], [1.0,1.0], [1.7,1.7], [2.5,2.5]],max_truth_boxes=30,thresh=0.6,nms_thresh=0.3):

  def _YOLOMetrics(y_true, y_pred, name=None):
      net_out = tf.transpose(y_pred, perm=[0, 2, 3, 1])

      _,h,w,c = net_out.get_shape().as_list()
      b = len(priors)
      anchors = np.array(priors)

      _probs, _confs, _coord, _areas, _upleft, _botright = tf.split(y_true, [num_classes,1,4,1,2,2], axis=3)
      _confs = tf.squeeze(_confs,3)
      _areas = tf.squeeze(_areas,3)

      net_out_reshape = tf.reshape(net_out, [-1, h, w, b, (4 + 1 + num_classes)])
      # Extract the coordinate prediction from net.out
      coords = net_out_reshape[:, :, :, :, :4]
      coords = tf.reshape(coords, [-1, h*w, b, 4])
      adjusted_coords_xy = logistic_activate_tensor(coords[:,:,:,0:2])
      adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, b, 2]) / np.reshape([w, h], [1, 1, 1, 2]))
      coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

      adjusted_c = logistic_activate_tensor(net_out_reshape[:, :, :, :, 4])
      adjusted_c = tf.reshape(adjusted_c, [-1, h*w, b, 1])

      adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
      adjusted_prob = tf.reshape(adjusted_prob, [-1, h*w, b, num_classes])

      adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

      wh = tf.pow(coords[:,:,:,2:4], 2) *  np.reshape([w, h], [1, 1, 1, 2])
      area_pred = wh[:,:,:,0] * wh[:,:,:,1]
      centers = coords[:,:,:,0:2]
      floor = centers - (wh * .5)
      ceil  = centers + (wh * .5)

      # calculate the intersection areas
      intersect_upleft   = tf.maximum(floor, _upleft)
      intersect_botright = tf.minimum(ceil, _botright)
      intersect_wh = intersect_botright - intersect_upleft
      intersect_wh = tf.maximum(intersect_wh, 0.0)
      intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

      # calculate the best IOU and metrics 
      iou = tf.truediv(intersect, _areas + area_pred - intersect)
      best_ious     = tf.reduce_max(iou, [2], True)
      recall        = tf.reduce_sum(tf.to_float(tf.greater(best_ious,0.5)), [1])
      sum_best_ious = tf.reduce_sum(best_ious, [1])
      gt_obj_areas  = tf.reduce_mean(_areas, [2], True)
      num_gt_obj    = tf.reduce_sum(tf.to_float(tf.greater(gt_obj_areas,tf.zeros_like(gt_obj_areas))), [1])
      avg_iou       = tf.truediv(sum_best_ious, num_gt_obj)
      avg_recall    = tf.truediv(recall, num_gt_obj)
 
      return {'avg_iou':tf.reduce_mean(avg_iou), 'avg_recall':tf.reduce_mean(avg_recall)}
  return _YOLOMetrics


