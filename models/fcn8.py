# Keras imports
from keras import layers
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.convolutional import (Conv2D, MaxPooling2D,
                                        ZeroPadding2D, Deconvolution2D)
from keras.layers.core import Dropout
from keras.regularizers import l2

# Custom layers import
from layers.ourlayers import (CropLayer2D, NdSoftmax)

from keras import backend as K
data_format = K.image_data_format()

# Paper: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
# Original caffe code: https://github.com/shelhamer/fcn.berkeleyvision.org
# Adapted from: MILA internal code


def build_fcn8(img_shape=(3, None, None), nclasses=8, l2_reg=0.,
               k_init='glorot_uniform', path_weights=None,
               freeze_layers_from=None):
    # Regularization warning
    if l2_reg > 0.:
        print ("Regularizing the weights: " + str(l2_reg))

    # Build network

    # CONTRACTING PATH

    # Input layer
    inputs = Input(img_shape)
    padded = ZeroPadding2D(padding=(100, 100), name='pad100')(inputs)

    # Block 1
    conv1_1 = Conv2D(64, (3,3), padding='valid', kernel_initializer=k_init,
                     activation='relu', name='conv1_1',
                     kernel_regularizer=l2(l2_reg))(padded)
    conv1_2 = Conv2D(64, (3,3), padding='same', kernel_initializer=k_init,
                     activation='relu', name='conv1_2',
                     kernel_regularizer=l2(l2_reg))(conv1_1)
    pool1 = MaxPooling2D((2, 2), (2, 2), name='pool1')(conv1_2)

    # Block 2
    conv2_1 = Conv2D(128, (3,3), padding='same', kernel_initializer=k_init,
                     activation='relu', name='conv2_1',
                     kernel_regularizer=l2(l2_reg))(pool1)
    conv2_2 = Conv2D(128, (3,3), padding='same', kernel_initializer=k_init,
                     activation='relu', name='conv2_2',
                     kernel_regularizer=l2(l2_reg))(conv2_1)
    pool2 = MaxPooling2D((2, 2), (2, 2), name='pool2')(conv2_2)

    # Block 3
    conv3_1 = Conv2D(256, (3,3), padding='same', kernel_initializer=k_init,
                     activation='relu', name='conv3_1',
                     kernel_regularizer=l2(l2_reg))(pool2)
    conv3_2 = Conv2D(256, (3,3), padding='same', kernel_initializer=k_init,
                     activation='relu', name='conv3_2',
                     kernel_regularizer=l2(l2_reg))(conv3_1)
    conv3_3 = Conv2D(256, (3,3), padding='same', kernel_initializer=k_init,
                     activation='relu', name='conv3_3',
                     kernel_regularizer=l2(l2_reg))(conv3_2)
    pool3 = MaxPooling2D((2, 2), (2, 2), name='pool3')(conv3_3)

    # Block 4
    conv4_1 = Conv2D(512, (3,3), padding='same', kernel_initializer=k_init,
                     activation='relu', name='conv4_1',
                     kernel_regularizer=l2(l2_reg))(pool3)
    conv4_2 = Conv2D(512, (3,3), padding='same', kernel_initializer=k_init,
                     activation='relu', name='conv4_2',
                     kernel_regularizer=l2(l2_reg))(conv4_1)
    conv4_3 = Conv2D(512, (3,3), padding='same', kernel_initializer=k_init,
                     activation='relu', name='conv4_3',
                     kernel_regularizer=l2(l2_reg))(conv4_2)
    pool4 = MaxPooling2D((2, 2), (2, 2), name='pool4')(conv4_3)

    # Block 5
    conv5_1 = Conv2D(512, (3,3), padding='same', kernel_initializer=k_init,
                     activation='relu', name='conv5_1',
                     kernel_regularizer=l2(l2_reg))(pool4)
    conv5_2 = Conv2D(512, (3,3), padding='same', kernel_initializer=k_init,
                     activation='relu', name='conv5_2',
                     kernel_regularizer=l2(l2_reg))(conv5_1)
    conv5_3 = Conv2D(512, (3,3), padding='same', kernel_initializer=k_init,
                     activation='relu', name='conv5_3',
                     kernel_regularizer=l2(l2_reg))(conv5_2)
    pool5 = MaxPooling2D((2, 2), (2, 2), name='pool5')(conv5_3)

    # Block 6 (fully conv)
    fc6 = Conv2D(4096, (7,7), padding='valid', kernel_initializer=k_init,
                 activation='relu', name='fc6',
                 kernel_regularizer=l2(l2_reg))(pool5)
    fc6 = Dropout(0.5)(fc6)

    # Block 7 (fully conv)
    fc7 = Conv2D(4096, (1,1), padding='valid', kernel_initializer=k_init,
                 activation='relu', name='fc7',
                 kernel_regularizer=l2(l2_reg))(fc6)
    fc7 = Dropout(0.5)(fc7)
    score_fr = Conv2D(nclasses, (1,1), padding='valid',
                      kernel_initializer=k_init, activation='relu',
                      name='score_fr')(fc7)

    # DECONTRACTING PATH
    # Unpool 1
    score_pool4 = Conv2D(nclasses, (1,1), padding='same',
                         kernel_initializer=k_init, activation='relu',
                         name='score_pool4',
                         kernel_regularizer=l2(l2_reg))(pool4)
    score2 = Deconvolution2D(nclasses, (4, 4), kernel_initializer=k_init,
                             activation='linear', padding='valid',
                             strides=(2, 2), name='score2',
                             kernel_regularizer=l2(l2_reg))(score_fr)
    score_pool4_crop = CropLayer2D(score2,
                                   name='score_pool4_crop')(score_pool4)
    score_fused = layers.add([score_pool4_crop, score2])

    # Unpool 2
    score_pool3 = Conv2D(nclasses, (1,1), padding='valid',
                         kernel_initializer=k_init, activation='relu',
                         name='score_pool3',
                         kernel_regularizer=l2(l2_reg))(pool3)
    score4 = Deconvolution2D(nclasses, (4, 4), kernel_initializer=k_init,
                             activation='linear', padding='valid',
                             strides=(2, 2), name='score4',
                             kernel_regularizer=l2(l2_reg),
                             use_bias=True)(score_fused)
    score_pool3_crop = CropLayer2D(score4, name='score_pool3_crop')(score_pool3)
    score_final = layers.add([score_pool3_crop, score4])


    upsample = Deconvolution2D(nclasses, (16, 16), kernel_initializer=k_init,
                               activation='linear', padding='valid',
                               strides=(8, 8), name='upsample',
                               kernel_regularizer=l2(l2_reg),
                               use_bias=False)(score_final)

    score = CropLayer2D(inputs, name='score')(upsample)

    # Softmax
    softmax_fcn8 = NdSoftmax()(score)

    # Complete model
    model = Model(inputs=inputs, outputs=softmax_fcn8)

    # Load pretrained Model
    if path_weights:
        load_matcovnet(model, n_classes=nclasses)

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
def load_matcovnet(model, path_weights='weights/pascal-fcn8s-dag.mat',
                   n_classes=11):

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
            if n_classes==21 or ('score' not in raw_name and \
               'upsample' not in raw_name and \
               'final' not in raw_name and \
               'probs' not in raw_name):

                print ('   Initializing weights of layer: ' + raw_name)
                print('    - Weights Loaded (FW x FH x FC x K): ' + str(param_value.shape))

                if data_format == 'channels_first':
                    # TH kernel shape: (depth, input_depth, rows, cols)
                    param_value = param_value.T
                    print('    - Weights Loaded (K x FC x FH x FW): ' + str(param_value.shape))
                else:
                    # TF kernel shape: (rows, cols, input_depth, depth)
                    param_value = param_value.transpose((1, 0, 2, 3))
                    print('    - Weights Loaded (FH x FW x FC x K): ' + str(param_value.shape))

                # Load current model weights
                w = model.get_layer(name=raw_name).get_weights()
                print('    - Weights model: ' + str(w[0].shape))
                if len(w)>1:
                    print('    - Bias model: ' + str(w[1].shape))

                print('    - Weights Loaded: ' + str(param_value.shape))
                w[0] = param_value
                model.get_layer(name=raw_name).set_weights(w)

        # Load bias terms
        if name.endswith(str_bias):
            raw_name = name[:-len(str_bias)]
            if n_classes==21 or ('score' not in raw_name and \
               'upsample' not in raw_name and \
               'final' not in raw_name and \
               'probs' not in raw_name):
                print ('   Initializing bias of layer: ' + raw_name)
                param_value = np.squeeze(param_value)
                w = model.get_layer(name=raw_name).get_weights()
                w[1] = param_value
                model.get_layer(name=raw_name).set_weights(w)
    return model


if __name__ == '__main__':
    input_shape = [3, 224, 224]
    print (' > Building')
    model = build_fcn8(input_shape, 11, 0.)
    print (' > Compiling')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()
