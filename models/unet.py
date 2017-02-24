"""
U-Net from https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix

Changes :
 - added init='glorot_uniform' to Convolution2D, Deconvolution2D
 - changed the final layers to get a number of channels equal to number of classes
   of a semantic segmentation problem.

Limitations:
 - number of rows and columns must be power of two (not necessarily square images)
 - tensorflow backend (why? because of Deconvolution2D ?)
 - need to provide the batch size as a parameter, but it can be None
"""

# Keras imports
from keras.models import Model
from keras.layers import (Input, merge)
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Dropout
from keras.regularizers import l2
from layers.deconv import Deconvolution2D
from layers.ourlayers import (CropLayer2D, NdSoftmax)


def build_unet(img_shape=(3, None, None), nclasses=8, l2_reg=0.,
               init='glorot_uniform', path_weights=None,
               freeze_layers_from=None, padding=100, dropout=True):

    # Regularization warning
    if l2_reg > 0.:
        print ("Regularizing the weights: " + str(l2_reg))

    # Input
    inputs = Input(img_shape, name='input')
    padded = ZeroPadding2D(padding=(padding, padding), name='padded')(inputs)

    # Block 1
    conv1_1 = Convolution2D(64, 3, 3, init, 'relu', border_mode='valid',
                            name='conv1_1', W_regularizer=l2(l2_reg))(padded)
    conv1_2 = Convolution2D(64, 3, 3, init, 'relu', border_mode='valid',
                            name='conv1_2', W_regularizer=l2(l2_reg))(conv1_1)
    pool1 = MaxPooling2D((2, 2), (2, 2), name='pool1')(conv1_2)

    # Block 2
    conv2_1 = Convolution2D(128, 3, 3, init, 'relu', border_mode='valid',
                            name='conv2_1', W_regularizer=l2(l2_reg))(pool1)
    conv2_2 = Convolution2D(128, 3, 3, init, 'relu', border_mode='valid',
                            name='conv2_2', W_regularizer=l2(l2_reg))(conv2_1)
    pool2 = MaxPooling2D((2, 2), (2, 2), name='pool2')(conv2_2)

    # Block 3
    conv3_1 = Convolution2D(256, 3, 3, init, 'relu', border_mode='valid',
                            name='conv3_1', W_regularizer=l2(l2_reg))(pool2)
    conv3_2 = Convolution2D(256, 3, 3, init, 'relu', border_mode='valid',
                            name='conv3_2', W_regularizer=l2(l2_reg))(conv3_1)
    pool3 = MaxPooling2D((2, 2), (2, 2), name='pool3')(conv3_2)

    # Block 4
    conv4_1 = Convolution2D(512, 3, 3, init, 'relu', border_mode='valid',
                            name='conv4_1', W_regularizer=l2(l2_reg))(pool3)
    conv4_2 = Convolution2D(512, 3, 3, init, 'relu', border_mode='valid',
                            name='conv4_2', W_regularizer=l2(l2_reg))(conv4_1)
    if dropout:
        conv4_2 = Dropout(0.5, name='drop1')(conv4_2)
    pool4 = MaxPooling2D((2, 2), (2, 2), name='pool4')(conv4_2)

    # Block 5
    conv5_1 = Convolution2D(1024, 3, 3, init, 'relu', border_mode='valid',
                            name='conv5_1', W_regularizer=l2(l2_reg))(pool4)
    conv5_2 = Convolution2D(1024, 3, 3, init, 'relu', border_mode='valid',
                            name='conv5_2', W_regularizer=l2(l2_reg))(conv5_1)
    if dropout:
        conv5_2 = Dropout(0.5, name='drop2')(conv5_2)
    # pool5 = MaxPooling2D((2, 2), (2, 2), name='pool4')(conv5_2)

    # Upsampling 1
    upconv4 = Deconvolution2D(512, 2, 2, conv5_2._keras_shape, init,
                              'linear', border_mode='valid', subsample=(2, 2),
                              name='upconv4', W_regularizer=l2(l2_reg))(conv5_2)
    conv4_2_crop = CropLayer2D(upconv4, name='conv4_2_crop')(conv4_2)
    upconv4_crop = CropLayer2D(upconv4, name='upconv4_crop')(upconv4)
    Concat_4 = merge([conv4_2_crop, upconv4_crop], mode='concat', concat_axis=3, name='Concat_4')
    conv6_1 = Convolution2D(512, 3, 3, init, 'relu', border_mode='valid',
                            name='conv6_1', W_regularizer=l2(l2_reg))(Concat_4)
    conv6_2 = Convolution2D(512, 3, 3, init, 'relu', border_mode='valid',
                            name='conv6_2', W_regularizer=l2(l2_reg))(conv6_1)

    # Upsampling 2
    upconv3 = Deconvolution2D(256, 2, 2, conv6_2._keras_shape, init,
                              'linear', border_mode='valid', subsample=(2, 2),
                              name='upconv3', W_regularizer=l2(l2_reg))(conv6_2)
    conv3_2_crop = CropLayer2D(upconv3, name='conv3_2_crop')(conv3_2)
    Concat_3 = merge([conv3_2_crop, upconv3], mode='concat', name='Concat_3')
    conv7_1 = Convolution2D(256, 3, 3, init, 'relu', border_mode='valid',
                            name='conv7_1', W_regularizer=l2(l2_reg))(Concat_3)
    conv7_2 = Convolution2D(256, 3, 3, init, 'relu', border_mode='valid',
                            name='conv7_2', W_regularizer=l2(l2_reg))(conv7_1)

    # Upsampling 3
    upconv2 = Deconvolution2D(128, 2, 2, conv7_2._keras_shape, init,
                              'linear', border_mode='valid', subsample=(2, 2),
                              name='upconv2', W_regularizer=l2(l2_reg))(conv7_2)
    conv2_2_crop = CropLayer2D(upconv2, name='conv2_2_crop')(conv2_2)
    Concat_2 = merge([conv2_2_crop, upconv2], mode='concat', name='Concat_2')
    conv8_1 = Convolution2D(128, 3, 3, init, 'relu', border_mode='valid',
                            name='conv8_1', W_regularizer=l2(l2_reg))(Concat_2)
    conv8_2 = Convolution2D(128, 3, 3, init, 'relu', border_mode='valid',
                            name='conv8_2', W_regularizer=l2(l2_reg))(conv8_1)

    # Upsampling 4
    upconv1 = Deconvolution2D(64, 2, 2, conv8_2._keras_shape, init,
                              'linear', border_mode='valid', subsample=(2, 2),
                              name='upconv1', W_regularizer=l2(l2_reg))(conv8_2)
    conv1_2_crop = CropLayer2D(upconv1, name='conv1_2_crop')(conv1_2)
    Concat_1 = merge([conv1_2_crop, upconv1], mode='concat', name='Concat_1')
    conv9_1 = Convolution2D(64, 3, 3, init, 'relu', border_mode='valid',
                            name='conv9_1', W_regularizer=l2(l2_reg))(Concat_1)
    conv9_2 = Convolution2D(64, 3, 3, init, 'relu', border_mode='valid',
                            name='conv9_2', W_regularizer=l2(l2_reg))(conv9_1)

    conv10 = Convolution2D(nclasses, 1, 1, init, 'linear', border_mode='valid',
                           name='conv10', W_regularizer=l2(l2_reg))(conv9_2)

    # Crop
    final_crop = CropLayer2D(inputs, name='final_crop')(conv10)

    # Softmax
    softmax_unet = NdSoftmax()(final_crop)

    # Complete model
    model = Model(input=inputs, output=softmax_unet)

    # Load pretrained Model
    if path_weights:
        pass

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
    print ('BUILD')
    model = build_unet(img_shape=(256, 512, 3), nclasses=11)
    print ('COMPILING')
    model.compile(loss="binary_crossentropy", optimizer="rmsprop")
    model.summary()
    print ('END COMPILING')
