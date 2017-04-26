# Python imports
import numpy as np

# Keras imports
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import (Convolution2D, AtrousConvolution2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout
from keras.layers import merge
from keras.regularizers import l2
from keras import initializers

# Custom layers import
from layers.ourlayers import (CropLayer2D, NdSoftmax)
from layers.deconv import Deconvolution2D
from initializations.initializations import bilinear_init
from tools.numpy2keras import load_numpy

# Keras dim orders
from keras import backend as K
dim_ordering = K.image_dim_ordering()
if dim_ordering == 'th':
    channel_idx = 1
else:
    channel_idx = 3


# Paper: https://arxiv.org/abs/1611.10080
# Original code in Mxnet: https://github.com/itijyou/ademxapp


# Create a convolution block: BN - RELU - CONV
def bn_relu_conv(inputs, n_filters, filter_size, stride, dropout,
                 use_bias, dilation=1, name=None, l2_reg=0.):

    first = BatchNormalization(mode=0, axis=channel_idx, name="bn"+name)(inputs)
    first = Activation('relu', name="res"+name+"_relu")(first)
    if dropout > 0:
        first = Dropout(dropout, name="res"+name+"_dropout")(first)

    if dilation == 1:
        second = Convolution2D(n_filters, filter_size, filter_size,
                               'he_normal', subsample=(stride, stride),
                               bias=use_bias, border_mode='same',
                               name="res"+name,
                               W_regularizer=l2(l2_reg))(first)
    else:
        second = AtrousConvolution2D(n_filters, filter_size, filter_size,
                                     'he_normal', subsample=(stride, stride),
                                     atrous_rate=(dilation, dilation),
                                     border_mode='same', bias=use_bias,
                                     name="res"+name,
                                     W_regularizer=l2(l2_reg))(first)

    return first, second


# Create a block
def create_block_n(inputs, n_filters, filter_size, strides, dropouts,
                   dilations, name, dim_match, plus_lvl, l2_reg=0.):
    # Parameters Initialization
    if dim_match:
        strides = [1]*len(strides)
        dilations0 = dilations[0]
    else:
        dilations0 = 1

    # Create residual block - Save the shortcut from the first one
    name = name+"_branch2"
    shortcut0, last_layer = bn_relu_conv(inputs, n_filters[0], filter_size[0],
                                         strides[0], dropouts[0], False,
                                         dilations0 if filter_size[0]>1 else 1,
                                         name=name+"a", l2_reg=l2_reg)

    for i in range(1, len(n_filters)):
        last_layer = bn_relu_conv(last_layer, n_filters[i], filter_size[i],
                                  strides[i], dropouts[i], False,
                                  dilations[i] if filter_size[i] > 1 else 1,
                                  name=name+"b"+str(i), l2_reg=l2_reg)[1]

    # Define shortcut
    if dim_match:
        shortcut = inputs
    else:
        shortcut_name = "res"+name[0:-1]+"1"
        shortcut = Convolution2D(n_filters[-1], 1, 1, 'he_normal',
                                 subsample=(strides[0], strides[0]),
                                 bias=False, name=shortcut_name,
                                 W_regularizer=l2(l2_reg))(shortcut0)

    # Fuse branches
    fused = merge([last_layer, shortcut], mode='sum',
                  name="a_plus"+str(plus_lvl))

    return fused


# Create the main body of the net
def create_body(inputs, blocks, n_filters, dilations, strides, dropouts,
                l2_reg=0.):

    # Create the first convolutional layer
    inputs = Convolution2D(64, 3, 3, bias=False, name="conv1a",
                           border_mode='same',
                           W_regularizer=l2(l2_reg))(inputs)

    # Create the next blocks
    cont = 0
    for i in range(len(blocks)):
        kernel_size = [3, 3] if len(n_filters[i]) == 2 else [1, 3, 1]
        for j in range(blocks[i]):
            name = str(i+1)+'a' if j == 0 else str(i+1)+'b'+str(j)
            inputs = create_block_n(inputs, n_filters[i], kernel_size,
                                    strides[i], dropouts[i], dilations[i],
                                    name=name, dim_match=(j > 0),
                                    plus_lvl=34+cont, l2_reg=l2_reg)
            cont = cont+1

    return inputs


# Create the classifier part
def create_classifier(body, data, n_classes, l2_reg=0.):
    # Include last layers
    top = BatchNormalization(mode=0, axis=channel_idx, name="bn7")(body)
    top = Activation('relu', name="relu7")(top)
    top = AtrousConvolution2D(512, 3, 3, 'he_normal', atrous_rate=(12, 12),
                              border_mode='same', name="conv6a",
                              W_regularizer=l2(l2_reg))(top)
    top = Activation('relu', name="conv6a_relu")(top)
    name = "hyperplane_num_cls_%d_branch_%d" % (n_classes, 12)

    def my_init(shape, name=None, dim_ordering='th'):
        return initializations.normal(shape, scale=0.01, name=name)
    top = AtrousConvolution2D(n_classes, 3, 3, my_init,
                              atrous_rate=(12, 12), border_mode='same',
                              name=name, W_regularizer=l2(l2_reg))(top)

    top = Deconvolution2D(n_classes, 16, 16, top._keras_shape, bilinear_init,
                          'linear', border_mode='valid', subsample=(8, 8),
                          bias=False, name="upscaling_"+str(n_classes),
                          W_regularizer=l2(l2_reg))(top)

    top = CropLayer2D(data, name='score')(top)
    top = NdSoftmax()(top)

    return top


# Create model of basic segnet
def deeplab_resnet38_aspp_ssm(inputs, n_classes, l2_reg=0.):
    # Number of bn_relu_convs in each block
    blocks = [0, 3, 3, 6, 3, 1, 1]
    # Number of filters in each bn_relu_convs
    n_filers = [[64, 64], [128, 128], [256, 256], [512, 512], [512, 1024],
                [512, 1024, 2048], [1024, 2048, 4096]]
    # Dilation parameter for each bn_relu_convs
    dilations = [[1, 1], [1, 1], [1, 1], [1, 1], [2, 2], [4, 4, 4], [4, 4, 4]]
    # Stride parameter for each bn_relu_convs
    strides = [[1, 1], [2, 1], [2, 1], [2, 1], [1, 1], [1, 1, 1], [1, 1, 1]]
    # Dropout parameter for each bn_relu_convs
    dropouts = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0,0,0.3], [0,0.3,0.5]]

    # Network Body - Only convolutionals blocks
    body = create_body(inputs, blocks, n_filers, dilations, strides,
                       dropouts, l2_reg)

    # Classifier
    top = create_classifier(body, inputs, n_classes, l2_reg)

    # Complete model
    model = Model(input=inputs, output=top)

    return model


def build_resnetFCN(img_shape=(3, None, None), n_classes=8, l2_reg=0.,
                    path_weights=None, freeze_layers_from=None):

    # Regularization warning
    if l2_reg > 0.:
        print ("   Regularizing the weights: " + str(l2_reg))

    # Input layer
    inputs = Input(img_shape)

    # Create model
    model = deeplab_resnet38_aspp_ssm(inputs, n_classes, l2_reg)

    # Load pretrained Model
    if path_weights:
        model = load_numpy(model, path_weights="weights/resnetFCN.npy")

    # Freeze some layers
    if freeze_layers_from is not None:
        freeze_layers(model, freeze_layers_from)

    return model


# Freeze layers for finetunning
def freeze_layers(model, freeze_layers_from):
    # Freeze the VGG part only
    if freeze_layers_from == 'base_model':
        print ('   Freezing base model layers')
        freeze_layers_from = 135

    # Show layers (Debug pruposes)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    print ('   Freezing from layer 0 to ' + str(freeze_layers_from))

    # Freeze layers
    for layer in model.layers[:freeze_layers_from]:
        layer.trainable = False
    for layer in model.layers[freeze_layers_from:]:
        layer.trainable = True


if __name__ == '__main__':
    input_shape = [3, 224, 224]
    print (' > Building')
    model = build_resnetFCN(input_shape, 11, 0.)

    print (' > Compiling')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()
