import time
import h5py

import theano
import numpy as np
import keras.backend as K
from keras.layers import Input, merge
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Dropout
from keras.models import Model, load_model
from keras.regularizers import l2
from layers.ourlayers import (CropLayer2D, NdSoftmax)
from layers.deconv import Deconvolution2D


def build_fcn8(img_shape,
               x_shape=None,
               dim_ordering='th',
               l2_reg=0.,
               nclasses=8,
               x_test_val=None,
               weights_file=False,
               **kwargs):

    # For Theano debug prouposes
    if x_test_val is not None:
        inputs.tag.test_value = x_test_val
        theano.config.compute_test_value = "warn"

    do = dim_ordering

    # Regularization warning
    if l2_reg > 0.:
        print ("Regularizing the weights: " + str(l2_reg))

    # Build network

    # CONTRACTING PATH

    # Input layer
    inputs = Input(img_shape)
    sh = inputs._keras_shape
    padded = ZeroPadding2D(padding=(100, 100), dim_ordering=do,
                           name='pad100')(inputs)

    # Block 1
    conv1_1 = Convolution2D(
           64, 3, 3, activation='relu', border_mode='valid', dim_ordering=do,
           name='conv1_1', W_regularizer=l2(l2_reg), trainable=True)(padded)
    conv1_2 = Convolution2D(
           64, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv1_2', W_regularizer=l2(l2_reg), trainable=True)(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering=do,
                         name='pool1')(conv1_2)

    # Block 2
    conv2_1 = Convolution2D(
           128, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv2_1', W_regularizer=l2(l2_reg), trainable=True)(pool1)
    conv2_2 = Convolution2D(
           128, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv2_2', W_regularizer=l2(l2_reg), trainable=True)(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering=do,
                         name='pool2')(conv2_2)

    # Block 3
    conv3_1 = Convolution2D(
           256, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv3_1', W_regularizer=l2(l2_reg), trainable=True
           )(pool2)
    conv3_2 = Convolution2D(
           256, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv3_2', W_regularizer=l2(l2_reg), trainable=True)(conv3_1)
    conv3_3 = Convolution2D(
           256, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv3_3', W_regularizer=l2(l2_reg), trainable=True)(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering=do,
                         name='pool3')(conv3_3)

    # Block 4
    conv4_1 = Convolution2D(
           512, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv4_1', W_regularizer=l2(l2_reg), trainable=True)(pool3)
    conv4_2 = Convolution2D(
           512, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv4_2', W_regularizer=l2(l2_reg), trainable=True)(conv4_1)
    conv4_3 = Convolution2D(
           512, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv4_3', W_regularizer=l2(l2_reg), trainable=True)(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering=do,
                         name='pool4')(conv4_3)

    # Block 5
    conv5_1 = Convolution2D(
           512, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv5_1', W_regularizer=l2(l2_reg), trainable=True)(pool4)
    conv5_2 = Convolution2D(
           512, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv5_2', W_regularizer=l2(l2_reg), trainable=True)(conv5_1)
    conv5_3 = Convolution2D(
           512, 3, 3, activation='relu', border_mode='same', dim_ordering=do,
           name='conv5_3', W_regularizer=l2(l2_reg), trainable=True)(conv5_2)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering=do,
                         name='pool5')(conv5_3)

    # Block 6 (fully conv)
    fc6 = Convolution2D(
          4096, 7, 7, activation='relu', border_mode='valid', dim_ordering=do,
          name='fc6', W_regularizer=l2(l2_reg), trainable=True)(pool5)
    fc6 = Dropout(0.5)(fc6)

    # Block 7 (fully conv)
    fc7 = Convolution2D(
          4096, 1, 1, activation='relu', border_mode='valid', dim_ordering=do,
          name='fc7', W_regularizer=l2(l2_reg), trainable=True)(fc6)
    fc7 = Dropout(0.5)(fc7)

    score_fr = Convolution2D(
          nclasses, 1, 1, activation='relu', border_mode='valid',
          dim_ordering=do, name='score_fr')(fc7)

    # DECONTRACTING PATH
    # Unpool 1
    score_pool4 = Convolution2D(
          nclasses, 1, 1, activation='relu', border_mode='same',
          dim_ordering=do, name='score_pool4', W_regularizer=l2(l2_reg),
          trainable=True)(pool4)
    score2 = Deconvolution2D(
        nb_filter=nclasses, nb_row=4, nb_col=4,
        input_shape=score_fr._keras_shape, subsample=(2, 2),
        border_mode='valid', activation='linear', W_regularizer=l2(l2_reg),
        dim_ordering=do, trainable=True, name='score2')(score_fr)
    score_pool4_crop = CropLayer2D(score2, dim_ordering=do,
                                   name='score_pool4_crop')(score_pool4)
    score_fused = merge([score_pool4_crop, score2], mode=custom_sum,
                        output_shape=custom_sum_shape, name='score_fused')

    # Unpool 2
    score_pool3 = Convolution2D(
        nclasses, 1, 1, activation='relu', border_mode='valid',
        dim_ordering=do, W_regularizer=l2(l2_reg),
        trainable=True, name='score_pool3')(pool3)
    score4 = Deconvolution2D(
        nb_filter=nclasses, nb_row=4, nb_col=4,
        input_shape=score_fused._keras_shape, subsample=(2, 2),
        border_mode='valid', activation='linear', W_regularizer=l2(l2_reg),
        dim_ordering=do, trainable=True, name='score4',
        bias=False)(score_fused)    # TODO: No bias??
    score_pool3_crop = CropLayer2D(score4, dim_ordering=do,
                                   name='score_pool3_crop')(score_pool3)
    score_final = merge([score_pool3_crop, score4], mode=custom_sum,
                        output_shape=custom_sum_shape, name='score_final')

    # Unpool 3
    upsample = Deconvolution2D(
        nb_filter=nclasses, nb_row=16, nb_col=16,
        input_shape=score_final._keras_shape, subsample=(8, 8),
        border_mode='valid', activation='linear', W_regularizer=l2(l2_reg),
        dim_ordering=do, trainable=True, name='upsample',
        bias=False)(score_final)  # TODO: No bias??
    score = CropLayer2D(inputs, dim_ordering=do, name='score')(upsample)

    # Softmax
    if do == 'th':
        softmax_fcn8 = NdSoftmax(1)(score)
    else:
        softmax_fcn8 = NdSoftmax(3)(score)

    # Complete model
    net = Model(input=inputs, output=softmax_fcn8)

    # Load weights
    if weights_file:
        print (' > Loading weights from pretrained model: ' + weights_file)
        net.load_weights(weights_file)

    return net


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
    print ('BUILD')
    # input_shape = [3, None, None]
    model = build_fcn8(input_shape,
                       # test_value=test_val,
                       x_shape=(batch, seq_len, ) + tuple(input_shape),
                       load_weights_fcn8=False,
                       seq_length=seq_len, nclasses=9)
    print ('COMPILING')
    model.compile(loss="binary_crossentropy", optimizer="rmsprop")
    model.summary()
    print ('END COMPILING')
