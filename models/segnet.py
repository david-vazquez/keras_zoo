# Keras imports
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D, UpSampling2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.regularizers import l2

# Custom layers import
from layers.ourlayers import (CropLayer2D, NdSoftmax, DePool2D)

# Paper: https://arxiv.org/abs/1511.00561
# Original code: https://github.com/alexgkendall/caffe-segnet
# Adapted from: https://github.com/imlab-uiip/keras-segnet
# Adapted from: https://github.com/pradyu1993/segnet


# Keras dim orders
def channel_idx():
    if K.image_dim_ordering() == 'th':
        return 1
    else:
        return 3


# Downsample blocks of the basic-segnet
def downsampling_block_basic(inputs, n_filters, filter_size,
                             W_regularizer=None):
    # This extra padding is used to prevent problems with different input
    # sizes. At the end the crop layer remove extra paddings
    pad = ZeroPadding2D(padding=(1, 1))(inputs)
    conv = Convolution2D(n_filters, filter_size, filter_size,
                         border_mode='same', W_regularizer=W_regularizer)(pad)
    bn = BatchNormalization(mode=0, axis=channel_idx())(conv)
    act = Activation('relu')(bn)
    maxp = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act)
    return maxp


# Upsample blocks of the basic-segnet
def upsampling_block_basic(inputs, n_filters, filter_size, unpool_layer=None,
                           W_regularizer=None, use_unpool=True):
    if use_unpool:
        up = DePool2D(unpool_layer)(inputs)
    else:
        up = UpSampling2D()(inputs)
    conv = Convolution2D(n_filters, filter_size, filter_size,
                         border_mode='same', W_regularizer=W_regularizer)(up)
    bn = BatchNormalization(mode=0, axis=channel_idx())(conv)
    return bn


# Create model of basic segnet
def build_segnet_basic(inputs, n_classes, depths=[64, 64, 64, 64],
                       filter_size=7, l2_reg=0.):
    """ encoding layers """
    enc1 = downsampling_block_basic(inputs, depths[0], filter_size, l2(l2_reg))
    enc2 = downsampling_block_basic(enc1, depths[1], filter_size, l2(l2_reg))
    enc3 = downsampling_block_basic(enc2, depths[2], filter_size, l2(l2_reg))
    enc4 = downsampling_block_basic(enc3, depths[3], filter_size, l2(l2_reg))

    """ decoding layers """
    dec1 = upsampling_block_basic(enc4, depths[3], filter_size, enc4,
                                  l2(l2_reg))
    dec2 = upsampling_block_basic(dec1, depths[2], filter_size, enc3,
                                  l2(l2_reg))
    dec3 = upsampling_block_basic(dec2, depths[1], filter_size, enc2,
                                  l2(l2_reg))
    dec4 = upsampling_block_basic(dec3, depths[0], filter_size, enc1,
                                  l2(l2_reg))

    """ logits """
    l1 = Convolution2D(n_classes, 1, 1, border_mode='valid')(dec4)
    score = CropLayer2D(inputs, name='score')(l1)
    softmax_segnet = NdSoftmax()(score)

    # Complete model
    model = Model(input=inputs, output=softmax_segnet)

    return model


# Downsampling block of the VGG
def downsampling_block_vgg(inputs, n_conv, n_filters, filter_size, layer_id,
                           l2_reg=None, activation='relu',
                           init='glorot_uniform', border_mode='same'):
    conv = ZeroPadding2D(padding=(1, 1))(inputs)
    for i in range(1, n_conv+1):
        name = 'conv' + str(layer_id) + '_' + str(i)
        conv = Convolution2D(n_filters, filter_size, filter_size, init,
                             border_mode=border_mode,
                             name=name,
                             W_regularizer=l2(l2_reg))(conv)
        conv = BatchNormalization(mode=0, axis=channel_idx(),
                                  name=name + '_bn')(conv)
        conv = Activation(activation, name=name + '_relu')(conv)
    conv = MaxPooling2D((2, 2), (2, 2), name='pool'+str(layer_id))(conv)
    return conv


# Upsampling block of the VGG
def upsampling_block_vgg(inputs, n_conv, n_filters, filter_size, layer_id,
                         l2_reg=None, unpool_layer=None, activation='relu',
                         init='glorot_uniform', border_mode='same',
                         use_unpool=True):
    if use_unpool:
        conv = DePool2D(unpool_layer, name='upsample'+str(layer_id))(inputs)
    else:
        conv = UpSampling2D()(inputs)
    for i in range(n_conv+1, 1, -1):
        conv = Convolution2D(n_filters, filter_size, filter_size, init,
                             border_mode=border_mode,
                             name='conv'+str(layer_id)+'_'+str(i)+'_D',
                             W_regularizer=l2(l2_reg))(conv)
        conv = BatchNormalization(mode=0, axis=channel_idx())(conv)
        conv = Activation(activation)(conv)
    return conv


# Create model of VGG Segnet
def build_segnet_vgg(inputs, n_classes, l2_reg=0.):

    """ encoding layers """
    enc1 = downsampling_block_vgg(inputs, 2, 64, 3, 1, l2_reg)
    enc2 = downsampling_block_vgg(enc1, 2, 128, 3, 2, l2_reg)
    enc3 = downsampling_block_vgg(enc2, 3, 256, 3, 3, l2_reg)
    enc4 = downsampling_block_vgg(enc3, 3, 512, 3, 4, l2_reg)
    enc5 = downsampling_block_vgg(enc4, 3, 512, 3, 5, l2_reg)

    """ decoding layers """
    dec5 = upsampling_block_vgg(enc5, 3, 512, 3, 5, l2_reg, enc5)
    dec4 = upsampling_block_vgg(dec5, 3, 512, 3, 4, l2_reg, enc4)
    dec3 = upsampling_block_vgg(dec4, 3, 256, 3, 3, l2_reg, enc3)
    dec2 = upsampling_block_vgg(dec3, 2, 128, 3, 2, l2_reg, enc2)
    dec1 = upsampling_block_vgg(dec2, 2, 64, 3, 1, l2_reg, enc1)

    """ logits """
    l1 = Convolution2D(n_classes, 1, 1, border_mode='valid')(dec1)
    score = CropLayer2D(inputs, name='score')(l1)
    softmax_segnet = NdSoftmax()(score)

    # Complete model
    model = Model(input=inputs, output=softmax_segnet)

    return model


def build_segnet(img_shape=(3, None, None), n_classes=8, l2_reg=0.,
                 init='glorot_uniform', path_weights=None,
                 freeze_layers_from=None, use_unpool=False, basic=False):

    # Regularization warning
    if l2_reg > 0.:
        print ("Regularizing the weights: " + str(l2_reg))

    # Input layer
    inputs = Input(img_shape)

    # Create basic Segnet
    if basic:
        model = build_segnet_basic(inputs, n_classes, [64, 64, 64, 64],
                                   7, l2_reg)
    else:
        model = build_segnet_vgg(inputs, n_classes, l2_reg)

    # Load pretrained Model
    if path_weights:
        load_matcovnet(model, path_weights, n_classes=n_classes)

    # Freeze some layers
    if freeze_layers_from is not None:
        freeze_layers(model, freeze_layers_from)

    return model


# Freeze layers for finetunning
def freeze_layers(model, freeze_layers_from):
    # Freeze the VGG part only
    if freeze_layers_from == 'base_model':
        print ('   Freezing base model layers')
        freeze_layers_from = 23

    # Show layers (Debug pruposes)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    print ('   Freezing from layer 0 to ' + str(freeze_layers_from))

    # Freeze layers
    for layer in model.layers[:freeze_layers_from]:
        layer.trainable = False
    for layer in model.layers[freeze_layers_from:]:
        layer.trainable = True


# Lad weights from matconvnet
def load_matcovnet(model, path_weights, n_classes):

    import scipy.io as sio
    import numpy as np

    print('   Loading pretrained model: ' + path_weights)
    # Depending the model has one name or other
    if 'tvg' in path_weights:
        str_filter = 'f'
        str_bias = 'b'
    else:
        str_filter = '_filter'
        str_bias = '_bias'

    # Open the .mat file in python
    W = sio.loadmat(path_weights)

    # Load the parameter values into the model
    num_params = W.get('params').shape[1]
    for i in range(num_params):
        # Get layer name from the saved model
        name = str(W.get('params')[0][i][0])[3:-2]

        # Get parameter value
        param_value = W.get('params')[0][i][1]

        # Load weights
        if name.endswith(str_filter):
            raw_name = name[:-len(str_filter)]

            # Skip final part
            if n_classes == 21 or ('score' not in raw_name and
                'upsample' not in raw_name and
                'final' not in raw_name and
                'probs' not in raw_name):

                print ('   Initializing weights of layer: ' + raw_name)
                print('    - Weights Loaded: ' + str(param_value.shape))
                param_value = param_value.T
                print('    - Weights Loaded: ' + str(param_value.shape))
                param_value = np.swapaxes(param_value, 2, 3)
                print('    - Weights Loaded: ' + str(param_value.shape))

                # Load current model weights
                w = model.get_layer(name=raw_name).get_weights()
                print('    - Weights model: ' + str(w[0].shape))
                if len(w) > 1:
                    print('    - Bias model: ' + str(w[1].shape))

                print('    - Weights Loaded: ' + str(param_value.shape))
                w[0] = param_value
                model.get_layer(name=raw_name).set_weights(w)

        # Load bias terms
        if name.endswith(str_bias):
            raw_name = name[:-len(str_bias)]
            if n_classes == 21 or ('score' not in raw_name and \
               'upsample' not in raw_name and \
               'final' not in raw_name and \
               'probs' not in raw_name):
                print ('Initializing bias of layer: ' + raw_name)
                param_value = np.squeeze(param_value)
                w = model.get_layer(name=raw_name).get_weights()
                w[1] = param_value
                model.get_layer(name=raw_name).set_weights(w)
    return model


if __name__ == '__main__':
    print ('BUILD full segnet')
    model_full = build_segnet(img_shape=(3, 360, 480), n_classes=8, l2_reg=0.,
                 init='glorot_uniform', path_weights=None,
                 freeze_layers_from=None, use_unpool=False, basic=False)
    print ('COMPILING full segnet')
    model_full.compile(loss="binary_crossentropy", optimizer="rmsprop")
    model_full.summary()
    print ('END COMPILING full segnet')

    print('')
    print ('BUILD basic segnet')
    model_basic = build_segnet(img_shape=(3, 360, 480), n_classes=8, l2_reg=0.,
                 init='glorot_uniform', path_weights=None,
                 freeze_layers_from=None, use_unpool=False, basic=True)
    print ('COMPILING basic segnet')
    model_basic.compile(loss="binary_crossentropy", optimizer="rmsprop")
    model_basic.summary()
    print ('END COMPILING basic segnet')
