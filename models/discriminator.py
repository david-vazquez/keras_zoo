from keras.models import Model
from keras.layers import Input, Dense, Activation, BatchNormalization, Reshape, LeakyReLU, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, UpSampling2D
# from keras.regularizers import l2


def build_discriminator(img_shape=(1, 28, 28), dropout_rate=0.25, l2_reg=0.):

    # Build Discriminative model ...
    d_input = Input(shape=img_shape)

    H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode='same',
                      activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode='same',
                      activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    H = Flatten()(H)
    H = Dense(256)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    d_output = Dense(2, activation='softmax')(H)

    model = Model(d_input, d_output)

    return model
