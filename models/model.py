from keras.models import Model
from keras.layers import (Input,
                          Activation,
                          merge,
                          Dense,
                          Flatten,
                          Dropout,
                          Permute,
                          Lambda,
                          merge)
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras import backend as K
from theano import tensor as T
import numpy as np
from .blocks import (bottleneck,
                     basic_block,
                     basic_block_mp,
                     residual_block,
                     crop_layer)

from keras.regularizers import l2

def softmax(x):
    '''
    Softmax that works on ND inputs.
    '''
    e = K.exp(x - K.max(x, axis=-1, keepdims=True))
    s = K.sum(e, axis=-1, keepdims=True)
    return e / s


def categorical_crossentropy_ND(y_true, y_pred):
    '''
    y_true must use an integer class representation
    y_pred must use a one-hot class representation
    '''
    shp_y_pred = K.shape(y_pred)
    y_pred_flat = K.reshape(y_pred, (K.prod(shp_y_pred[:-1]), shp_y_pred[-1]))
    y_true_flat = K.flatten(y_true)
    y_true_flat = K.cast(y_true_flat, 'int32')
    out = K.categorical_crossentropy(y_pred_flat, y_true_flat)
    return K.mean(out)


def cce_with_regional_penalty(weight, power, nb_row, nb_col):
    '''
    y_true has shape [batch_size, 1, w, h]
    y_pred has shape [batch_size, num_classes, w, h]
    '''
    def f(y_true, y_pred):
        loss = categorical_crossentropy_ND(y_true, y_pred)
        
        y_true_flat = y_true.flatten()
        y_true_flat = K.cast(y_true_flat, 'int32')
        y_true_onehot_flat = T.extra_ops.to_one_hot(y_true_flat,
                                                    nb_class=y_pred.shape[-1])
        y_true_onehot = K.reshape(y_true_onehot_flat, y_pred.shape)
        
        abs_err = K.abs(y_true_onehot-y_pred)
        abs_err = K.permute_dimensions(abs_err, [0,3,1,2])
        kernel = T.ones((2, 2, nb_row, nb_col)) / np.float32(nb_row*nb_col)
        conv = K.conv2d(abs_err, kernel, strides=(1, 1), border_mode='same')
        penalty = K.pow(conv, power)
        return (1-weight)*loss + weight*K.mean(penalty)
    return f

    
class layer_tracker(object):
    '''
    Helper object to keep track of previously added layer and allow layer
    retrieval by name from a dictionary.
    '''
    def __init__(self):
        self.layer_dict = {}
        self.prev_layer = None
        
    def record(self, layer, name):
        layer.name = name
        self.layer_dict[name] = layer
        self.prev_layer = layer
        
    def __getitem__(self, name):
        return self.layer_dict[name]
    
    
def make_long_skip(prev_layer, concat_layer, num_concat_filters,
                   num_target_filters, use_skip_blocks, repetitions,
                   dropout, skip, batch_norm, W_regularizer,
                   merge_mode='concat', block=bottleneck):
    '''
    Helper function to create a long skip connection with concatenation.
    Concatenated information is not transformed if use_skip_blocks is False.
    '''
    if use_skip_blocks:
        concat_layer = residual_block(block, nb_filter=num_concat_filters,
                                      repetitions=repetitions, dropout=dropout,
                                      skip=skip, batch_norm=batch_norm,
                                      W_regularizer=l2(W_regularizer))(concat_layer)
    if merge_mode == 'sum':
        if prev_layer._keras_shape[1] != num_target_filters:
            prev_layer = Convolution2D(num_target_filters, 1, 1,
                                       init='he_normal', border_mode='valid',
                                       W_regularizer=l2(W_regularizer))(prev_layer)
        if concat_layer._keras_shape[1] != num_target_filters:
            concat_layer = Convolution2D(num_target_filters, 1, 1,
                                     init='he_normal', border_mode='valid',
                                     W_regularizer=l2(W_regularizer))(concat_layer)
    prev_layer = crop_layer(output_dim=K.shape(concat_layer)[2:])(prev_layer)
    concat = merge([prev_layer, concat_layer], mode=merge_mode, concat_axis=1)
    return concat
    
    
def assemble_model(input_shape, num_classes, num_main_blocks, main_block_depth,
                   num_init_blocks, input_num_filters, short_skip=True,
                   long_skip=True, long_skip_merge_mode='concat',
                   mainblock=None, initblock=None, use_skip_blocks=True,
                   skipblock=None, relative_num_across_filters=1, dropout=0.,
                   batch_norm=True, W_regularizer=None):
    
    '''
    By default, use depth 2 bottleneck for mainblock
    '''
    if mainblock is None:
        mainblock = bottleneck
    if initblock is None:
        initblock = basic_block_mp
    if skipblock is None:
        skipblock = basic_block_mp
    
    '''
    main_block_depth can be a list per block or a single value 
    -- ensure the list length is correct (if list) and that no length is 0
    '''
    if not hasattr(main_block_depth, '__len__'):
        if main_block_depth==0:
            raise ValueError("main_block_depth must never be zero")
    else:
        if len(main_block_depth)!=num_main_blocks+1:
            raise ValueError("main_block_depth must have " 
                             "`num_main_blocks+1` values when " 
                             "passed as a list")
        for d in main_block_depth:
            if d==0:
                raise ValueError("main_block_depth must never be zero")
    
    '''
    Returns the depth of a mainblock for a given pooling level
    '''
    def get_repetitions(level):
        if hasattr(main_block_depth, '__len__'):
            return main_block_depth[level]
        return main_block_depth
    
    '''
    Constant kwargs passed to the init and main blocks.
    '''
    block_kwargs = {'skip': short_skip,
                    'dropout': dropout,
                    'batch_norm': batch_norm,
                    'W_regularizer': W_regularizer}
    
    '''
    If long skip is not (the defualt) identity, always pass these
    parameters to make_long_skip
    '''
    long_skip_kwargs = {'use_skip_blocks': use_skip_blocks,
                        'repetitions': 1,
                        'merge_mode': long_skip_merge_mode,
                        'block': skipblock}
    long_skip_kwargs.update(block_kwargs)
    
    layers = layer_tracker()
    
    # INPUT
    layers.record(Input(shape=input_shape), name='input')
    
    # Initial convolution
    layers.record(Convolution2D(input_num_filters, 3, 3,
                               init='he_normal', border_mode='same',
                               W_regularizer=l2(W_regularizer))(layers.prev_layer),
                  name='first_conv')
    
    # DOWN (initial subsampling blocks)
    for b in range(num_init_blocks):
        layers.record(initblock(input_num_filters, subsample=True,
                                **block_kwargs)(layers.prev_layer),
                      name='initblock_d'+str(b))
        print("INIT DOWN {}: {} -- {}".format(b, layers.prev_layer.name,
                                              layers.prev_layer._keras_shape))
    
    # DOWN (resnet blocks)
    for b in range(num_main_blocks):
        num_filters = input_num_filters*(2**b)
        layers.record(residual_block(mainblock, nb_filter=num_filters, 
                              repetitions=get_repetitions(b), subsample=True,
                              **block_kwargs)(layers.prev_layer),
                      name='mainblock_d'+str(b))
        print("MAIN DOWN {}: {} (depth {}) -- {}".format(b,
              layers.prev_layer.name, get_repetitions(b),
              layers.prev_layer._keras_shape))
        
    # ACROSS
    num_filters = input_num_filters*(2**num_main_blocks)
    num_filters *= relative_num_across_filters
    layers.record(residual_block(mainblock, nb_filter=num_filters, 
                                 repetitions=get_repetitions(num_main_blocks),
                                 subsample=True, upsample=True,
                                 **block_kwargs)(layers.prev_layer), 
                  name='mainblock_a')
    print("ACROSS: {} (depth {}) -- {}".format( \
          layers.prev_layer.name, get_repetitions(num_main_blocks),
          layers.prev_layer._keras_shape))

    # UP (resnet blocks)
    for b in range(num_main_blocks-1, -1, -1):
        num_filters = input_num_filters*(2**b)
        if long_skip:
            num_across_filters = num_filters*relative_num_across_filters
            repetitions = get_repetitions(num_main_blocks)
            layers.record(make_long_skip(prev_layer=layers.prev_layer,
                                     concat_layer=layers['mainblock_d'+str(b)],
                                     num_concat_filters=num_across_filters,
                                     num_target_filters=num_filters,
                                     **long_skip_kwargs),
                          name='concat_main_'+str(b))
        layers.record(residual_block(mainblock, nb_filter=num_filters, 
                               repetitions=get_repetitions(b), upsample=True,
                               **block_kwargs)(layers.prev_layer),
                      name='mainblock_u'+str(b))
        print("MAIN UP {}: {} (depth {}) -- {}".format(b,
              layers.prev_layer.name, get_repetitions(b),
              layers.prev_layer._keras_shape))
        
    # UP (final upsampling blocks)
    for b in range(num_init_blocks-1, -1, -1):
        if long_skip:
            num_across_filters = input_num_filters*relative_num_across_filters
            repetitions = get_repetitions(num_main_blocks)
            layers.record(make_long_skip(prev_layer=layers.prev_layer,
                                     concat_layer=layers['initblock_d'+str(b)],
                                     num_concat_filters=num_across_filters,
                                     num_target_filters=input_num_filters,
                                     **long_skip_kwargs),
                          name='concat_init_'+str(b))    
        layers.record(initblock(input_num_filters, upsample=True,
                                **block_kwargs)(layers.prev_layer),
                      name='initblock_u'+str(b))
        print("INIT UP {}: {} -- {}".format(b,
              layers.prev_layer.name, layers.prev_layer._keras_shape))
        
    # Final convolution
    layers.record(Convolution2D(input_num_filters, 3, 3,
                               init='he_normal', border_mode='same',
                               W_regularizer=l2(W_regularizer))(layers.prev_layer),
                  name='final_conv')
    if long_skip:
        num_across_filters = input_num_filters*relative_num_across_filters
        repetitions = get_repetitions(num_main_blocks)
        layers.record(make_long_skip(prev_layer=layers.prev_layer,
                                     concat_layer=layers['first_conv'],
                                     num_concat_filters=num_across_filters,
                                     num_target_filters=input_num_filters,
                                     **long_skip_kwargs),
                      name='concat_top')
    if batch_norm:
        layers.record(BatchNormalization(mode=0, axis=1)(layers.prev_layer),
                    name='final_bn')
    layers.record(Activation('relu')(layers.prev_layer), name='final_relu')
    
    # OUTPUT (SOFTMAX)
    layers.record(Convolution2D(num_classes,1,1,activation='linear', 
                  W_regularizer=l2(W_regularizer))(layers.prev_layer), name='sm_1')
    layers.record(Permute((2,3,1))(layers.prev_layer), name='sm_2')
    layers.record(Activation(softmax)(layers.prev_layer), name='softmax')
    layers.record(Permute((3,1,2))(layers.prev_layer), name='output')
    
    # MODEL
    model = Model(input=layers['input'], output=layers['output'])

    return model
