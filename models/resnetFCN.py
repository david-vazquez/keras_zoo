# Keras imports
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import (Convolution2D, AtrousConvolution2D,
                                        MaxPooling2D, ZeroPadding2D,
                                        UpSampling2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout
from keras.layers import merge
from keras.regularizers import l2
from layers.deconv import Deconvolution2D
#from keras.engine.topology import Merge

# Custom layers import
from layers.ourlayers import (CropLayer2D, NdSoftmax, DePool2D)

# Keras dim orders
from keras import backend as K
dim_ordering = K.image_dim_ordering()
if dim_ordering == 'th':
    channel_idx = 1
else:
    channel_idx = 3


# Create a convolution block: BN - RELU - CONV
def bn_relu_conv(inputs, n_filters, filter_size, stride, pading, dropout,
                 use_bias, dilation=1, name=None, names=[]):

    # BN
    bn_name = "bn" + name
    bn = BatchNormalization(mode=0, axis=channel_idx, name=bn_name)(inputs)

    # Relu
    relu_name = "res%s_relu" % name if name is not None else names[3]
    relu = Activation('relu', name=relu_name)(bn)

    # Dropout
    if dropout>0:
        dropout_name = "res%s_dropout" % name if name is not None else names[3]
        relu = Dropout(dropout, name=dropout_name)(relu)

    # Conv
    conv_name = "res%s" % name if name is not None else names[0]
    pad = ZeroPadding2D(padding=(pading, pading))(relu)
    if dilation==1:
        conv = Convolution2D(n_filters, filter_size, filter_size,
                             subsample=(stride, stride), bias=use_bias,
                             name=conv_name)(pad)
    else:
        conv = AtrousConvolution2D(n_filters, filter_size, filter_size,
                                   subsample=(stride, stride),
                                   atrous_rate=(dilation, dilation),
                                   bias=use_bias, name=conv_name)(pad)

    return relu, conv


def create_block_n(inputs, n_filters, filter_size, strides, paddings, dropouts,
                   dilations, sub_name, level_id, dim_match, plus_lvl):
    # Parameters Initialization
    if dim_match:
        # Change the stride to 1 for the others iterations
        strides=[1]*len(strides) # TODO: Python use this variable by reference. It is changed outside the function!!
        dilations0=dilations[0]
        paddings0=paddings[0]
    else:
        dilations0=1
        paddings0=1

    # Create residual block - Save the shortcut from the first one
    name = sub_name + "_branch2"
    shortcut0, last_layer = bn_relu_conv(inputs, n_filters[0], filter_size[0],
        strides[0], paddings0 if filter_size[0]>1 else 0,
        dropouts[0], False, dilations0 if filter_size[0]>1 else 1,
        name=name+"a")

    for i in range(1, len(n_filters)):
        name_conv = name + "b" + str(i)
        last_layer = bn_relu_conv(last_layer, n_filters[i], filter_size[i],
            strides[i], paddings[i] if filter_size[i]>1 else 0, dropouts[i],
            False, dilations[i] if filter_size[i]>1 else 1,
            name=name_conv)[1]

    # Define shortcut
    if dim_match:
        shortcut=inputs
    else:
        shortcut_name = "res"+name[0:-1]+"1"
        shortcut = Convolution2D(n_filters[-1], 1, 1,
                                 subsample=(strides[0], strides[0]), bias=False,
                                 name=shortcut_name)(shortcut0)

    # Fuse branches
    fused = merge([last_layer, shortcut], mode='sum',
                  name="a_plus"+str(plus_lvl))

    return fused


# Create the main body of the net
def create_body(inputs, vBlocks, vvFilter, vvDilation, vvStride, vvPad,
                vvDropout):

    # Parameters Initialization
    cont=0

    # Loop for each block
    for level_id, blocks in enumerate(vBlocks):
        filter_id = level_id+1

        # Number of repetitions of each block
        for block_id in range(blocks):
            print ('level_id: ' + str(level_id))
            print ('filter_id: ' + str(filter_id))
            print ('block_id: ' + str(block_id))

            # Define layer name
            if block_id == 0:
                sub_name = "%d%c" % (level_id + 1, chr(ord('a')))
            else:
                sub_name = "%d%c%d" % (level_id + 1, 'b', block_id)

            # Assign kernel size
            if len(vvFilter[filter_id])==2:
                kernel_size=[3,3]
            elif len(vvFilter[filter_id])==3:
                kernel_size=[1,3,1]

            # Create block
            inputs = create_block_n(inputs, vvFilter[filter_id], kernel_size,
                                    vvStride[level_id], vvPad[level_id],
                                    vvDropout[level_id], vvDilation[level_id],
                                    sub_name=sub_name, level_id=level_id,
                                    dim_match=block_id>0, plus_lvl=34+cont)

            cont=cont+1

    return inputs


def create_classifier(body, data, dummy_patch_size, n_classes,
                      include_labels, ignore_label):

    # Include last layers
    top = BatchNormalization(mode=0, axis=channel_idx, name="bn7")(body)
    top = Activation('relu', name="relu7")(top)
    top = ZeroPadding2D(padding=(12, 12))(top)
    top = AtrousConvolution2D(512, 3, 3, atrous_rate=(12,12), name="conv6a")(top)
    top = Activation('relu', name="conv6a_relu")(top)

    name = "hyperplane_num_cls_%d_branch_%d" % (n_classes, 12)
    top = ZeroPadding2D(padding=(12, 12))(top)
    top = AtrousConvolution2D(n_classes, 3, 3, atrous_rate=(12,12), name=name)(top)
    # weight_filler=dict(type="gaussian", std=0.01),
    # bias_filler=dict(type="constant", value=0.0),
    #tops.append(conv)

    name = "upscaling_%d" % n_classes
    top = Deconvolution2D(n_classes, 16, 16, top._keras_shape, 'glorot_uniform',
                          'linear', border_mode='valid', subsample=(8, 8),
                          bias=False, name=name)(top)
    top = CropLayer2D(data, name='score')(top)
    top = NdSoftmax()(top)

    return top
    #
    # if include_labels:
    #     label_blob = net_spec.set_layer(NetworkFactory.get_label_name(0),
    #                                     L.Input(shape=dict(dim=[1, dummy_patch_size, dummy_patch_size]),
    #                                             include=dict(phase=caffe.TRAIN)))
    #
    #     net_spec.set_layer(NetworkFactory.get_loss_name(0),
    #                        L.AdaptiveSoftmaxWithLoss(crop_blob,
    #                                          label_blob,
    #                                          softmax_param=dict(num_selected=87025,p=1.3,frac=0.25),
    #                                          include=dict(phase=caffe.TRAIN),
    #                                          loss_param=dict(normalize=True,
    #                                                          ignore_label=ignore_label)))


# Create model of basic segnet
def deeplab_resnet38_aspp_ssm(inputs, n_classes, include_labels=True, ignore_label=255):

    # Parameters initialization
    vBlocks = [0,3,3,6,3,1,1]     #repetitions of each block
    vvFilter = [[64],[64,64],[128,128],[256,256],[512,512],[512,1024],[512,1024,2048],[1024,2048,4096]] # vvFilter = [[entry], [block1-1,block1-2], [block2-1,block2-2],...,[blockn-1,blockn-m]]
    vvDilation = [[1,1],[1,1],[1,1],[1,1],[2,2],[4,4,4],[4,4,4]] # vvDilation = [[block1-1,block1-2], [block2-1,block2-2],...,[blockn-1,blockn-m]]
    vvStride = [[1,1],[2,1],[2,1],[2,1],[1,1],[1,1,1],[1,1,1]] # vvStride = [[block1-1,block1-2], [block2-1,block2-2],...,[blockn-1,blockn-m]]
    vvPad = [[1,1],[1,1],[1,1],[1,1],[2,2],[4,4,4],[4,4,4]] # vvPad = [[block1-1,block1-2], [block2-1,block2-2],...,[blockn-1,blockn-m]]
    vvDropout = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0,0.3],[0,0.3,0.5]] # vvPad = [[block1-1,block1-2], [block2-1,block2-2],...,[blockn-1,blockn-m]]
    dummy_patch_size = 50 # TODO: change to 500
    num_labels = n_classes

    # Input Data
    conv1a = Convolution2D(64, 3, 3, bias=False,
                             name="conv1a", border_mode='same')(inputs)



    # Network Body - Only convolutionals blocks
    body = create_body(conv1a, vBlocks, vvFilter, vvDilation, vvStride, vvPad,
                       vvDropout)

    # Classifier
    top = create_classifier(body, inputs, dummy_patch_size, num_labels,
                            include_labels, ignore_label)

    # Complete model
    model = Model(input=inputs, output=top)

    return model


def build_resnetFCN(img_shape=(3, None, None), n_classes=8, l2_reg=0.,
                 init='glorot_uniform', path_weights=None,
                 freeze_layers_from=None):

    # Regularization warning
    if l2_reg > 0.:
        print ("Regularizing the weights: " + str(l2_reg))

    # Input layer
    inputs = Input(img_shape)

    # Create model
    model = deeplab_resnet38_aspp_ssm(inputs, n_classes)

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
