# Keras imports
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Dropout
from keras.regularizers import l2

# Custom layers import
from layers.ourlayers import (CropLayer2D, NdSoftmax)
from layers.deconv import Deconvolution2D


def build_fcn8(img_shape=(3, None, None), nclasses=8, l2_reg=0.,
               init='glorot_uniform'):

    # Regularization warning
    if l2_reg > 0.:
        print ("Regularizing the weights: " + str(l2_reg))

    # Build network

    # CONTRACTING PATH

    # Input layer
    inputs = Input(img_shape)
    padded = ZeroPadding2D(padding=(100, 100), name='pad100')(inputs)

    # Block 1
    conv1_1 = Convolution2D(64, 3, 3, init, 'relu', border_mode='valid',
                            name='conv1_1', W_regularizer=l2(l2_reg))(padded)
    conv1_2 = Convolution2D(64, 3, 3, init, 'relu', border_mode='same',
                            name='conv1_2', W_regularizer=l2(l2_reg))(conv1_1)
    pool1 = MaxPooling2D((2, 2), (2, 2), name='pool1')(conv1_2)

    # Block 2
    conv2_1 = Convolution2D(128, 3, 3, init, 'relu', border_mode='same',
                            name='conv2_1', W_regularizer=l2(l2_reg))(pool1)
    conv2_2 = Convolution2D(128, 3, 3, init, 'relu', border_mode='same',
                            name='conv2_2', W_regularizer=l2(l2_reg))(conv2_1)
    pool2 = MaxPooling2D((2, 2), (2, 2), name='pool2')(conv2_2)

    # Block 3
    conv3_1 = Convolution2D(256, 3, 3, init, 'relu', border_mode='same',
                            name='conv3_1', W_regularizer=l2(l2_reg))(pool2)
    conv3_2 = Convolution2D(256, 3, 3, init, 'relu', border_mode='same',
                            name='conv3_2', W_regularizer=l2(l2_reg))(conv3_1)
    conv3_3 = Convolution2D(256, 3, 3, init, 'relu', border_mode='same',
                            name='conv3_3', W_regularizer=l2(l2_reg))(conv3_2)
    pool3 = MaxPooling2D((2, 2), (2, 2), name='pool3')(conv3_3)

    # Block 4
    conv4_1 = Convolution2D(512, 3, 3, init, 'relu', border_mode='same',
                            name='conv4_1', W_regularizer=l2(l2_reg))(pool3)
    conv4_2 = Convolution2D(512, 3, 3, init, 'relu', border_mode='same',
                            name='conv4_2', W_regularizer=l2(l2_reg))(conv4_1)
    conv4_3 = Convolution2D(512, 3, 3, init, 'relu', border_mode='same',
                            name='conv4_3', W_regularizer=l2(l2_reg))(conv4_2)
    pool4 = MaxPooling2D((2, 2), (2, 2), name='pool4')(conv4_3)

    # Block 5
    conv5_1 = Convolution2D(512, 3, 3, init, 'relu', border_mode='same',
                            name='conv5_1', W_regularizer=l2(l2_reg))(pool4)
    conv5_2 = Convolution2D(512, 3, 3, init, 'relu', border_mode='same',
                            name='conv5_2', W_regularizer=l2(l2_reg))(conv5_1)
    conv5_3 = Convolution2D(512, 3, 3, init, 'relu', border_mode='same',
                            name='conv5_3', W_regularizer=l2(l2_reg))(conv5_2)
    pool5 = MaxPooling2D((2, 2), (2, 2), name='pool5')(conv5_3)

    # Block 6 (fully conv)
    fc6 = Convolution2D(4096, 7, 7, init, 'relu', border_mode='valid',
                        name='fc6', W_regularizer=l2(l2_reg))(pool5)
    fc6 = Dropout(0.5)(fc6)

    # Block 7 (fully conv)
    fc7 = Convolution2D(4096, 1, 1, init, 'relu', border_mode='valid',
                        name='fc7', W_regularizer=l2(l2_reg), )(fc6)
    fc7 = Dropout(0.5)(fc7)

    score_fr = Convolution2D(nclasses, 1, 1, init, 'relu',
                             border_mode='valid', name='score_fr')(fc7)

    # DECONTRACTING PATH
    # Unpool 1
    score_pool4 = Convolution2D(nclasses, 1, 1, init, 'relu', border_mode='same',
                                name='score_pool4',
                                W_regularizer=l2(l2_reg))(pool4)
    score2 = Deconvolution2D(nclasses, 4, 4, score_fr._keras_shape, init,
                             'linear', border_mode='valid', subsample=(2, 2),
                             name='score2', W_regularizer=l2(l2_reg))(score_fr)


    score_pool4_crop = CropLayer2D(score2,
                                   name='score_pool4_crop')(score_pool4)
    score_fused = merge([score_pool4_crop, score2], mode=custom_sum,
                        output_shape=custom_sum_shape, name='score_fused')

    # Unpool 2
    score_pool3 = Convolution2D(nclasses, 1, 1, init, 'relu', border_mode='valid',
                                name='score_pool3',
                                W_regularizer=l2(l2_reg))(pool3)

    score4 = Deconvolution2D(nclasses, 4, 4, score_fused._keras_shape, init,
                             'linear', border_mode='valid', subsample=(2, 2),
                             bias=False,     # TODO: No bias??
                             name='score4', W_regularizer=l2(l2_reg))(score_fused)

    score_pool3_crop = CropLayer2D(score4, name='score_pool3_crop')(score_pool3)
    score_final = merge([score_pool3_crop, score4], mode=custom_sum,
                        output_shape=custom_sum_shape, name='score_final')

    upsample = Deconvolution2D(nclasses, 16, 16, score_final._keras_shape, init,
                               'linear', border_mode='valid', subsample=(8, 8),
                               bias=False,     # TODO: No bias??
                               name='upsample', W_regularizer=l2(l2_reg))(score_final)

    score = CropLayer2D(inputs, name='score')(upsample)

    # Softmax
    softmax_fcn8 = NdSoftmax()(score)

    # Complete model
    net = Model(input=inputs, output=softmax_fcn8)

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
