from keras.models import Model
from keras.layers import Input, Dense, Activation, BatchNormalization, Reshape
from keras.layers.convolutional import Convolution2D, UpSampling2D
import keras.backend as K
# from keras.regularizers import l2


def build_generator(img_shape=[100], n_channels=200, l2_reg=0.):

    # Build Generative model ...
    g_input = Input(shape=img_shape)
    H = Dense(n_channels*14*14, init='glorot_normal')(g_input)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    if K.image_dim_ordering() == 'th':
        H = Reshape([n_channels, 14, 14])(H)
    else:
        H = Reshape([14, 14, n_channels])(H)
    H = UpSampling2D(size=(2, 2))(H)

    H = Convolution2D(n_channels/2, 3, 3, border_mode='same',
                      init='glorot_uniform')(H)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)

    H = Convolution2D(n_channels/4, 3, 3, border_mode='same',
                      init='glorot_uniform')(H)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)

    H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
    g_output = Activation('sigmoid')(H)

    model = Model(g_input, g_output)

    return model
