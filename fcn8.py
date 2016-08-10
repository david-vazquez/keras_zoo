import time
import h5py
import numpy as np

import theano

import keras.backend as K
from keras.layers import Input, merge
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D, Deconvolution2D)
from keras.layers.core import Dropout
from keras.models import Model
from keras.regularizers import l2

from layers.ourlayers import CropLayer2D, MergeSequences, NdSoftmax


def build_fcn8(img_shape,
               x_shape=None,
               dim_ordering='th',
               regularize_weights=False,
               dropout=False,
               nclasses=8,
               x_test_val=None,
               load_weights=False,
               **kwargs):

    do = dim_ordering

    batch_size = x_shape[0]
    seq_length = x_shape[1]
    input_shape = (seq_length, )+tuple(img_shape)

    inputs = Input(shape=input_shape, batch_shape=x_shape)

    if x_test_val is not None:
        inputs.tag.test_value = x_test_val
        theano.config.compute_test_value = "warn"

    sh = inputs._keras_shape

    if regularize_weights:
        print "regularizing the weights"
        l2_reg = l2(0.001)
    else:
        l2_reg = None

    # Build network

    # CONTRACTING PATH
    flat = MergeSequences(merge=True, batch_size=batch_size,
                          name='flat')(inputs)
    padded = ZeroPadding2D(padding=(100, 100), dim_ordering=do,
                           name='pad100')(flat)

    # Block 1
    conv1_1 = Convolution2D(
           64, 3, 3, activation='relu', border_mode='valid', dim_ordering=do,
           name='conv1_1', W_regularizer=l2_reg, trainable=True)(padded)
    conv1_2 = Convolution2D(
           64, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv1_2', W_regularizer=l2_reg, trainable=True)(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering=do,
                         name='pool1')(conv1_2)

    # Block 2
    conv2_1 = Convolution2D(
           128, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv2_1', W_regularizer=l2_reg, trainable=True)(pool1)
    conv2_2 = Convolution2D(
           128, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv2_2', W_regularizer=l2_reg, trainable=True)(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering=do,
                         name='pool2')(conv2_2)

    # Block 3
    conv3_1 = Convolution2D(
           256, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv3_1', W_regularizer=l2_reg, trainable=True
           )(pool2)
    conv3_2 = Convolution2D(
           256, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv3_2', W_regularizer=l2_reg, trainable=True)(conv3_1)
    conv3_3 = Convolution2D(
           256, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv3_3', W_regularizer=l2_reg, trainable=True)(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering=do,
                         name='pool3')(conv3_3)

    # Block 4
    conv4_1 = Convolution2D(
           512, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv4_1', W_regularizer=l2_reg, trainable=True)(pool3)
    conv4_2 = Convolution2D(
           512, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv4_2', W_regularizer=l2_reg, trainable=True)(conv4_1)
    conv4_3 = Convolution2D(
           512, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv4_3', W_regularizer=l2_reg, trainable=True)(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering=do,
                         name='pool4')(conv4_3)

    # Block 5
    conv5_1 = Convolution2D(
           512, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv5_1', W_regularizer=l2_reg, trainable=True)(pool4)
    conv5_2 = Convolution2D(
           512, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv5_2', W_regularizer=l2_reg, trainable=True)(conv5_1)
    conv5_3 = Convolution2D(
           512, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv5_3', W_regularizer=l2_reg, trainable=True)(conv5_2)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering=do,
                         name='pool5')(conv5_3)

    # Block 6 (fully conv)
    fc6 = Convolution2D(
          4096, 7, 7, activation='relu', border_mode='valid', dim_ordering=do,
          name='fc6', W_regularizer=l2_reg, trainable=True)(pool5)

    fc6 = Dropout(0.5)(fc6)

    fc7 = Convolution2D(
          4096, 1, 1, activation='relu', border_mode='valid', dim_ordering=do,
          name='fc7', W_regularizer=l2_reg, trainable=True)(fc6)

    fc7 = Dropout(0.5)(fc7)

    score_fr = Convolution2D(
          nclasses, 1, 1, activation='relu', border_mode='valid',
          dim_ordering=do, name='score_fr')(fc7)

    # DECONTRACTING PATH

    # Unpool 1
    score_pool4 = Convolution2D(
          nclasses, 1, 1, activation='relu', border_mode='same',
          dim_ordering=do, W_regularizer=l2_reg,
          trainable=True, name='score_pool4')(pool4)
    
    score2 = Deconvolution2D(
        nb_filter=nclasses, nb_row=4, nb_col=4,
        output_shape=score_pool4._keras_shape, subsample=(2, 2),
        border_mode='valid', activation='linear', W_regularizer=l2_reg,
        dim_ordering=do, trainable=True, name='score2')(score_fr)
    score_pool4_crop = CropLayer2D(score2._keras_shape, 
                                   dim_ordering=do,
                                   name='score_pool4_crop')(score_pool4)
    

    score_fused = merge([score_pool4_crop, score2], mode=custom_sum,
                        output_shape=custom_sum_shape, name='score_fused')

    # Unpool 2
    score_pool3 = Convolution2D(
        nclasses, 1, 1, activation='relu', border_mode='valid',
        dim_ordering=do, W_regularizer=l2_reg,
        trainable=True, name='score_pool3')(pool3)

    score4 = Deconvolution2D(
        nb_filter=nclasses, nb_row=4, nb_col=4,
        output_shape=score_pool3._keras_shape, subsample=(2, 2),
        border_mode='valid', activation='linear', W_regularizer=l2_reg,
        bias=False, dim_ordering=do,
        trainable=True, name='score4')(score_fused)

    score_pool3_crop = CropLayer2D(score4._keras_shape, dim_ordering=do,
                                   name='score_pool3_crop')(score_pool3)
    score_final = merge([score_pool3_crop, score4], mode=custom_sum,
                        output_shape=custom_sum_shape, name='score_final')

    # Unpool 3
    upsample = Deconvolution2D(
        nb_filter=nclasses, nb_row=16, nb_col=16,
        output_shape=inputs._keras_shape, subsample=(8, 8),
        border_mode='valid', activation='linear', dim_ordering=do,
        W_regularizer=l2_reg,
        trainable=True, name='upsample', bias=False)(score_final)

    score = CropLayer2D(sh, dim_ordering=do, name='score')(upsample)

    if do == 'th':
        softmax_fcn8 = NdSoftmax(1)(score)
    else:
        softmax_fcn8 = NdSoftmax(3)(score)

    deflat = MergeSequences(merge=False, batch_size=batch_size,
                            name='deflat')(softmax_fcn8)
    net = Model(input=inputs, output=deflat)

    # Load weights
    if load_weights:
        load_weights(net, filepath=load_weights)

    return net


def load_weights(model, filepath):
    f = h5py.File(filepath, mode='r')

    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    # we batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        if len(weight_names):
            weight_values = [g[weight_name] for weight_name in weight_names]
            layer = model.get_layer(name=name)
            symbolic_weights = layer.trainable_weights + \
                layer.non_trainable_weights
            if len(weight_values) != len(symbolic_weights):
                raise Exception('Layer #' + str(k) +
                                ' (named "' + layer.name +
                                '" in the current model) was found to '
                                'correspond to layer ' + name +
                                ' in the save file. '
                                'However the new layer ' + layer.name +
                                ' expects ' + str(len(symbolic_weights)) +
                                ' weights, but the saved weights have ' +
                                str(len(weight_values)) +
                                ' elements.')
            weight_value_tuples += zip(symbolic_weights, weight_values)
    K.batch_set_value(weight_value_tuples)
    f.close()
    print("weights from filepath loaded")


def custom_sum(tensors):
    t1, t2 = tensors
    return t1 + t2


def custom_sum_shape(tensors):
    t1, t2 = tensors
    return t1

if __name__ == '__main__':
    start = time.time()
    input_shape = [2, 224, 224]
    seq_len = 1
    batch = 1
    print 'BUILD'
    # input_shape = [3, None, None]
    model = build_fcn8(input_shape,
                       # test_value=test_val,
                       batch_size=batch,
                       x_shape=(batch, seq_len, ) + tuple(input_shape),
                       load_weights_fcn8=False,
                       decontrating_path='recurrent',
                       seq_length=seq_len, nclasses=9)
    print 'COMPILING'
    model.compile(loss="binary_crossentropy", optimizer="rmsprop")
    model.summary()
    print 'END COMPILING'
