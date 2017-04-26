# Keras imports
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import (Convolution2D, AtrousConvolution2D,
                                        MaxPooling2D, ZeroPadding2D,
                                        UpSampling2D, Conv2DTranspose)
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import merge
from keras.layers.merge import Concatenate
from layers.ourlayers import (CropLayer2D, NdSoftmax)
from keras.regularizers import l2
from layers.deconv import Deconvolution2D
from keras import initializers
import sys
sys.setrecursionlimit(10000)

# Paper: https://arxiv.org/abs/1608.06993
# Original code: https://github.com/liuzhuang13/DenseNet
# Adapted from: https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet
# Adapted from: https://github.com/robertomest/convnet-study
# Adapted from: https://github.com/titu1994/DenseNet
# Adapted from: https://github.com/SimJeg/FC-DenseNet

# Custom layers import
from layers.ourlayers import (CropLayer2D, NdSoftmax, DePool2D)

# Keras dim orders
from keras import backend as K
data_format = K.image_data_format()
if data_format == 'channels_first':
    channel_idx = 1
else:
    channel_idx = 3


# Initial convolution used for ImageNet: 7x7Conv + MaxPool
def initial_conv_7x7(inputs, n_filters, weight_decay=1e-4):
    l = Convolution2D(n_filters, (7, 7), kernel_initializer="he_uniform", strides=(2, 2),
                      use_bias=False, padding="same", name="initial_conv",
                      kernel_regularizer=l2(weight_decay))(inputs)
    l = MaxPooling2D((3, 3), (2, 2), border_mode='same', name='initial_pool')(l)
    return l, n_filters


# Initial convolution used for Cifar10, Cifar100 and SVHN: 3x3Conv
def initial_conv_3x3(inputs, n_filters, weight_decay=1e-4):
    l = Convolution2D(n_filters, (3, 3), kernel_initializer="he_uniform",
                      use_bias=False, padding="same", name="initial_conv",
                      kernel_regularizer=l2(weight_decay))(inputs)
    return l, n_filters


# Create a basic convolution layer: BN - Relu - Conv - Dropout
def bn_relu_conv(inputs, n_filters, filter_size, dropout=0.,
                 weight_decay=1e-4, name=''):

    l = BatchNormalization(axis=channel_idx,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay),
                           name=name+'_bn')(inputs)
    l = Activation('relu', name=name+'_relu')(l)

    l = Convolution2D(n_filters, (filter_size, filter_size), kernel_initializer='he_uniform',
                      use_bias=False, padding='same',
                      kernel_regularizer=l2(weight_decay), name=name+'_conv')(l)

    if dropout>0.:
        l = Dropout(dropout, name=name+'_dropout')(l)

    return l


# Create a dense block: N layers of (BN - Relu - Conv - Dropout)
# densely connected
def dense_block(inputs, n_layers, n_filters, growth_rate, dropout=None,
                use_bottleneck=False, weight_decay=1e-4, name='',return_concat_list=False):
    list_feat = [inputs]
    for i in range(n_layers):
        # Add bottleneck to reduce input dimension
        if use_bottleneck:
            inputs = bn_relu_conv(inputs, growth_rate*4, 1, dropout,
                                  weight_decay,
                                  name=name+'_layer'+str(i)+'botneck')
        # Get layer output
        inputs = bn_relu_conv(inputs, growth_rate, 3, dropout, weight_decay,
                              name=name+'_layer'+str(i))
        list_feat.append(inputs)
        # Concatenate input with layer ouput
        inputs = Concatenate([inputs, list_feat],name=name+'_layer'+str(i)+'_merge')
        n_filters += growth_rate
    if return_concat_list:
        return inputs, n_filters, list_feat
    else:
        return inputs, n_filters
# Create a transition down: BN - Relu - Conv - Dropout - Pooling
def transition_down(inputs, n_filters, compression=1.0, dropout=0.,
                    weight_decay=1e-4, name=''):
    n_filters = int(n_filters*compression)
    l = bn_relu_conv(inputs, n_filters, 1, dropout, name=name+'_TD')
    l = AveragePooling2D((2, 2), strides=(2, 2), name=name+'_TD_pool')(l)
    return l, n_filters
    
# Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection
def transition_up(skip_connection, block_to_upsample, n_filters_keep):
    # Upsample
    l = Concatenate(block_to_upsample)
    l = Conv2DTranspose(n_filters_keep, (3,3), strides=(2,2), padding="same",
                        kernel_initializer="he_uniform", activation="relu")
    # Concatenate with skip connection
    l = Concatenate([l, skip_connection])
    return l


# Build the DenseNet model
def DenseNet(inputs, n_classes, n_dense_block, n_layers, growth_rate,
             n_initial_filters, compression=1.0, use_bottleneck=False,
             dropout=None, weight_decay=1e-4, initial_conv=3, top=False):
             
    block_to_upsample = []
    # Initial convolution
    if initial_conv==3:
        l, n_filters = initial_conv_3x3(inputs, n_initial_filters,
                                        weight_decay)
    elif initial_conv==7:
        l, n_filters = initial_conv_7x7(inputs, n_initial_filters,
                                        weight_decay)
    else:
        raise ValueError('Unknown initial convolution')

    skip_list = []
    # Add dense blocks
    for block_idx in range(n_dense_block):
        # Add dense block
        l, n_filters, block_to_upsample = dense_block(l, n_layers[block_idx], n_filters,
                                   growth_rate, dropout, use_bottleneck, 
                                   weight_decay, name='block'+str(block_idx), return_concat_list=True)
        skip_list.append(l)
        # Add transition down except for the last block
        if block_idx<(n_dense_block - 1):
            l, n_filters = transition_down(l, n_filters, compression, dropout,
                                           weight_decay,
                                           name='block'+str(block_idx))
    skip_list = skip_list[::-1] # reverse the skip list
    
    # Add classifier at the end
    if top:
        l = BatchNormalization(axis=channel_idx,
                               gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay),
                               name='classifier_bn')(l)
        l = Activation('relu', name='classifier_relu')(l)
        l = GlobalAveragePooling2D(name='classifier_pool')(l)
        l = Dense(n_classes,
                  activation='softmax',
                  W_regularizer=l2(weight_decay),
                  b_regularizer=l2(weight_decay),
                  name='classifier_dense')(l)
    
    model = Model(inputs=[inputs], outputs=[l], name="DenseNet")
    return model, skip_list, n_filters


# Build the densenet model
def build_densenetFCN(img_shape=(3, None, None), n_classes=8, l2_reg=0.,
                      path_weights=None, freeze_layers_from=None):

    # Regularization warning
    if l2_reg > 0.:
        print ("Regularizing the weights: " + str(l2_reg))

    # Input layer
    inputs = Input(img_shape)

    # Create model
    if False:
        # Cifar10 model
        # Parameters
        compression = 0.5 # 0.5 or 1.0
        use_bottleneck = True # True or False
        growth_rate = 12 # 12 or 24
        depth = 100 # 40 or 100

        # Compute the number of layers per dense block
        assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"
        n_layers = int((depth-4) / 3)
        if use_bottleneck:
            n_layers = int(n_layers // 2)
        n_layers = [n_layers]*3

        # Build the model
        model = DenseNet(inputs, n_classes, n_dense_block=3,
                         n_layers=n_layers, growth_rate=growth_rate,
                         n_initial_filters=16, compression=compression,
                         use_bottleneck=use_bottleneck, dropout=0.2,
                         weight_decay=l2_reg, initial_conv=3)
    else:
        # ImageNet model
        # Parameters
        compression = 0.5 # 0.5 or 1.0
        use_bottleneck = True # True or False
        growth_rate = 32 # 32 or 48
        n_dense_block = 4
        n_layers = [6, 12, 24, 16, 0, 0, 0, 0, 0]
        skip_list = []
        # Compute the number of layers per dense block
        if type(n_layers) == list:
            assert (len(n_layers) == 2 * n_dense_block + 1)
        elif type(n_layers) == int:
            n_layers = [n_layers] * (2 * n_dense_block + 1)
        else:
            raise ValueError
        print "HOLA"
        model, skip_list, n_filters = DenseNet(inputs, n_classes, n_dense_block,
                         n_layers=n_layers, growth_rate=growth_rate,
                         n_initial_filters=2*growth_rate,
                         compression=compression,
                         dropout=0.2, weight_decay=l2_reg, initial_conv=7)
        # The last dense_block does not have a transition_down_block
        # return the concatenated feature maps without the concatenation of the input
        x = model.layers[-1].output
        print "HOLA"
        _, nb_filter, concat_list = dense_block(x, n_layers[-1], n_filters, growth_rate, dropout=None,
                                    use_bottleneck=False, weight_decay=1e-4, name='s',return_concat_list=True)
        for block in range(n_dense_block):
            n_filters_keep = growth_rate * n_layers[n_dense_block + block]
            x = Concatenate(concat_list[1:])
            t = transition_up(skip_list[block], concat_list, n_filters_keep)
            x = Concatenate([t, skip_list[block]])
            print "HOLA"
            _, nb_filter, concat_list = dense_block(x, n_layers[block], n_filters, growth_rate, dropout=None,
                                    use_bottleneck=False, weight_decay=1e-4, name='up', return_concat_list=True)
        #softmax
        l1 = Convolution2D(n_classes, (1, 1), padding='valid')(x)
        score = CropLayer2D(inputs, name='score')(l1)
        softmax_segnet = NdSoftmax()(score)
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


if __name__ == '__main__':
    input_shape = [3, 224, 224]
    print (' > Building')
    model = build_resnetFCN(input_shape, 11, 0.)

    print (' > Compiling')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()
