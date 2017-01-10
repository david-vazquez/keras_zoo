from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D


def build_lenet(img_shape=(1, 32, 32),
                dim_ordering='th',
                l2_reg=0.,
                weights_file=False,
                n_classes=10,
                **kwargs):

    # initialize the model
    model = Sequential()

    # first set of CONV => RELU => POOL
    model.add(Convolution2D(20, 5, 5, border_mode="same", input_shape=img_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Convolution2D(50, 5, 5, border_mode="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

    if weights_file:
        model.load_weights(weights_file)

    return model
