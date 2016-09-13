from keras.layers import (Activation,
                          merge,
                          Dropout,
                          Lambda)
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import (Convolution2D,
                                        MaxPooling2D,
                                        UpSampling2D)
from keras import backend as K
from keras.regularizers import l2
from keras.layers.core import Layer

class crop_layer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(crop_layer, self).__init__(**kwargs)
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], None,
                None)
    def call(self, x, mask=None):
        return x[:, :, :self.output_dim[0],
                :self.output_dim[1]]

# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=False, upsample=False,
                  batch_norm=True, W_regularizer=None):
    
    def f(input):
        processed = input
        if batch_norm:
            processed = BatchNormalization(mode=0, axis=1)(processed)
        processed = Activation('relu')(processed)
        stride = (1, 1)
        if subsample:
            stride = (2, 2)
        if upsample:
            processed = UpSampling2D(size=(2, 2))(processed)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col,
                             subsample=stride, init='he_normal',
                             border_mode='same',
                             W_regularizer=l2(W_regularizer))(processed)

    return f


# Adds a shortcut between input and residual block and merges them with 'sum'
def _shortcut(input, residual, W_regularizer=None, subsample=False,
              upsample=False):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]
    
    shortcut = input
    
    # Downsample input
    if subsample:
           
        w = 2
        h = 2
        def downsample_output_shape(input_shape):
            output_shape = list(input_shape)
            output_shape[-2] = None if output_shape[-2]==None else output_shape[-2] // w
            output_shape[-1] = None if output_shape[-1]==None else output_shape[-1] // h
            return tuple(output_shape)
        shortcut = Lambda(lambda x: x[:,:, ::w, ::h],
                          output_shape=downsample_output_shape)(input)
        
    # Upsample input
    elif upsample:
             
        w = 2
        h = 2
        shortcut = UpSampling2D(size=(w, h))(input)
        
    # Adjust input channels to match residual
    if not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1],
                                 nb_row=1, nb_col=1,
                                 init='he_normal', border_mode='valid',
                                 W_regularizer=l2(W_regularizer))(shortcut)
        
    return merge([shortcut, residual], mode='sum')


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filter * 4
def bottleneck(nb_filter, subsample=False, upsample=False, skip=True,
               dropout=0., batch_norm=True, W_regularizer=None):
    def f(input):
        processed = _bn_relu_conv(nb_filter, 1, 1,
                                 subsample=subsample, batch_norm=batch_norm,
                                 W_regularizer=W_regularizer)(input)
        processed = _bn_relu_conv(nb_filter, 3, 3, batch_norm=batch_norm,
                                 W_regularizer=W_regularizer)(processed)
        processed = _bn_relu_conv(nb_filter * 4, 1, 1,
                                 upsample=upsample, batch_norm=batch_norm,
                                 W_regularizer=W_regularizer)(processed)
        if dropout > 0:
            processed = Dropout(dropout)(processed)
            
        output = processed
        if skip:
            output = _shortcut(input, output, W_regularizer=W_regularizer,
                               subsample=subsample, upsample=upsample)
        return output

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def basic_block(nb_filter, subsample=False, upsample=False, skip=True,
                dropout=0., batch_norm=True, W_regularizer=None):
    def f(input):
        processed = _bn_relu_conv(nb_filter, 3, 3,
                                  subsample=subsample, batch_norm=batch_norm,
                                  W_regularizer=W_regularizer)(input)
        if dropout > 0:
            processed = Dropout(dropout)(processed)
        processed = _bn_relu_conv(nb_filter, 3, 3,
                                  upsample=upsample, batch_norm=batch_norm,
                                  W_regularizer=W_regularizer)(processed)
        
        output = processed
        if skip:
            output = _shortcut(input, processed, W_regularizer=W_regularizer,
                               subsample=subsample, upsample=upsample)
        return output

    return f


# Builds a residual block with repeating bottleneck blocks.
def residual_block(block_function, nb_filter, repetitions, skip=True,
                   dropout=0., subsample=False, upsample=False,
                   batch_norm=True, W_regularizer=None):
    def f(input):
        for i in range(repetitions):
            kwargs = {'nb_filter': nb_filter, 'skip': skip, 'dropout': dropout,
                      'subsample': False, 'upsample': False,
                      'batch_norm': batch_norm, 'W_regularizer': W_regularizer}
            if i==0:
                kwargs['subsample'] = subsample
            if i==repetitions-1:
                kwargs['upsample'] = upsample
            input = block_function(**kwargs)(input)
        return input

    return f 


# A single basic 3x3 convolution
def basic_block_mp(nb_filter, subsample=False, upsample=False, skip=True,
                   dropout=0., batch_norm=True, W_regularizer=None):
    def f(input):
        processed = input
        if batch_norm:
            processed = BatchNormalization(mode=0, axis=1)(processed)
        processed = Activation('relu')(processed)
        if subsample:
            processed = MaxPooling2D(pool_size=(2,2))(processed)
        processed = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3,
                                  init='he_normal', border_mode='same',
                                  W_regularizer=l2(W_regularizer))(processed)
        if dropout > 0:
            processed = Dropout(dropout)(processed)
        if upsample:
            processed = UpSampling2D(size=(2, 2))(processed)
            
        output = processed
        if skip:
            output = _shortcut(input, processed, W_regularizer=W_regularizer,
                               subsample=subsample, upsample=upsample)
        return output
    
    return f
