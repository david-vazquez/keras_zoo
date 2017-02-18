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

TODO: add L2 regularization ?
"""
import numpy as np

# Keras imports
from keras.models import Model
from keras.layers import (Input, merge)
from keras.layers.convolutional import (Convolution2D, Deconvolution2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import (Activation, Dropout)
# from keras.regularizers import l2
# TODO add regularization

from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU

from layers.ourlayers import (CropLayer2D, NdSoftmax)


def conv_block(x, f, name, bn_mode, bn_axis, bn=True, subsample=(2, 2)):
    x = LeakyReLU(0.2)(x)
    x = Convolution2D(f, 3, 3, subsample=subsample, name=name,
                      init='glorot_uniform', border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)

    return x


def deconv_block(x, x2, f, h, w, batch_size, name, bn_mode, bn_axis,
                 bn=True, dropout=False):

    o_shape = (batch_size, h * 2, w * 2, f)
    x = Activation("relu")(x)
    x = Deconvolution2D(f, 3, 3, output_shape=o_shape, subsample=(2, 2),
                        init='glorot_uniform', border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = merge([x, x2], mode='concat', concat_axis=bn_axis)

    return x


# States if a number is a power of two
def is_power2(num):
    return num > 0 and ((num & (num - 1)) == 0)


def generator_unet_deconv(img_dim, n_classes, bn_mode, batch_size,
                          model_name="unet"):

    n_filters = 64
    bn_axis = -1
    h, w, nb_channels = img_dim
    min_s = min(img_dim[:-1])  # TODO: ??

    # TODO: Solve this
    assert K.backend() == "tensorflow", "Not implemented with theano backend"

    # TODO: Remove this
    assert is_power2(h) and is_power2(w), "rows and cols must be powers of 2"

    unet_input = Input(shape=img_dim, name="input")

    # Prepare encoder filters
    n_conv = int(np.floor(np.log(min_s) / np.log(2)))
    list_n_filters = [n_filters * min(8, (2 ** i)) for i in range(n_conv)]

    # Encoder first layer
    list_encoder = [Convolution2D(list_n_filters[0], 3, 3, subsample=(2, 2),
                                  name="conv2D_1",
                                  border_mode="same")(unet_input)]
    h, w = h / 2, w / 2

    # Encoder next layers
    for i, f in enumerate(list_n_filters[1:]):
        conv = conv_block(list_encoder[-1], f, "conv2D_%s" % (i + 2),
                          bn_mode, bn_axis)
        list_encoder.append(conv)
        h, w = h / 2, w / 2

    # Prepare decoder filters
    list_n_filters = list_n_filters[:-1][::-1]
    if len(list_n_filters) < n_conv - 1:
        list_n_filters.append(n_filters)

    # Decoder first layer
    list_decoder = [deconv_block(list_encoder[-1], list_encoder[-2],
                                 list_n_filters[0], h, w, batch_size,
                                 "upconv2D_1", bn_mode, bn_axis, dropout=True)]
    h, w = h * 2, w * 2

    # Decoder next layers
    for i, f in enumerate(list_n_filters[1:]):
        conv = deconv_block(list_decoder[-1], list_encoder[-(i + 3)], f, h,
                            w, batch_size, "upconv2D_%s" % (i + 2),
                            bn_mode, bn_axis, dropout=(i < 2))
        list_decoder.append(conv)
        h, w = h * 2, w * 2

    x = Activation("relu")(list_decoder[-1])
    # The original implementation outputs a new image of same shape as input image
    # goes as:
    #    o_shape = (batch_size,) + img_dim
    #    x = Deconvolution2D(nb_channels, 3, 3, output_shape=o_shape, subsample=(2, 2), border_mode="same")(x)
    #    x = Activation("tanh")(x)
    #    generator_unet = Model(input=unet_input, output=x)

    o_shape = (batch_size, img_dim[0], img_dim[1], n_classes)
    x = Deconvolution2D(n_classes, 3, 3, output_shape=o_shape,
                        subsample=(2, 2), border_mode="same")(x)

    score = CropLayer2D(unet_input, name='score')(x)
    # CropLayer2D is probably innecessary since x has already the same rows and
    # cols than unet_input

    softmx = NdSoftmax()(score)
    # TODO: add a parameter to choose between sigmoid, softmax and tanh ?
    # In another implementation, http://vess2ret.inesctec.pt they do so.
    # or return score = logits ?

    generator_unet = Model(input=unet_input, output=softmx)

    return generator_unet


# All network models (vgg, segnet etc) have a built_net() function called by
# make() method of Model_Factory class, that return a Keras model.
def build_unet(img_shape, n_classes, bn_mode=2, batch_size=None, l2_reg=0.,
               path_weights=None):

    # Create model
    model = generator_unet_deconv(img_shape, n_classes, bn_mode, batch_size,
                                  model_name="unet")

    # Load pretrained Model
    if path_weights:
        pass

    return model


if __name__ == '__main__':
    print ('BUILD')
    model = build_unet(img_shape=(256, 512, 3), n_classes=11)
    print ('COMPILING')
    model.compile(loss="binary_crossentropy", optimizer="rmsprop")
    model.summary()
    print ('END COMPILING')
