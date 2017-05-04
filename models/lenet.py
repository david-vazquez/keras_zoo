from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2

# paper: yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
def build_lenet(img_shape=(1, 28, 28), n_classes=10, l2_reg=0.):

    # initialize the model
    model = Sequential()

    # first set of CONV => RELU => POOL
    model.add(Convolution2D(20, 5, 5, border_mode="same",
                            input_shape=img_shape, W_regularizer=l2(l2_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Convolution2D(50, 5, 5, border_mode="same",
                            W_regularizer=l2(l2_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500, W_regularizer=l2(l2_reg)))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(n_classes, W_regularizer=l2(l2_reg)))
    model.add(Activation("softmax"))

    return model
