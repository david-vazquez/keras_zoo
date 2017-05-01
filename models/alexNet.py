from keras.layers.core import Lambda
from keras import backend as K
from keras import layers
from keras.models import Model
from keras.layers import Input, merge, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2

# Paper: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
# Caffe code: https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# Adapted from: https://github.com/heuritech/convnets-keras


def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """

    def f(X):
        if K.image_data_format() == 'channels_last':
            b, r, c, ch = X.get_shape()
        else:
            b, ch, r, c = X.shape

        half = n // 2
        square = K.square(X)
        scale = k
        if K.image_data_format() == 'channels_first':
            extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 2, 3, 1)), (0, half))
            extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
            for i in range(n):
                scale += alpha * extra_channels[:, i:i+ch, :, :]
        elif K.image_data_format() == 'channels_last':
            extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 3, 1, 2)), padding=((half, half), (0, 0)))
            extra_channels = K.permute_dimensions(extra_channels, (0, 2, 3, 1))
            for i in range(n):
                scale += alpha * extra_channels[:, :, :, i:i+int(ch)]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        div = K.shape(X)[axis] // ratio_split

        if axis == 0:
            output = X[id_split*div:(id_split+1)*div, :, :, :]
        elif axis == 1:
            output = X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:, :, id_split*div:(id_split+1)*div, :]
        elif axis == 3:
            output = X[:, :, :, id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")

        return output

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)


def build_alexNet(img_shape=(3, 227, 227), n_classes=1000, l2_reg=0.):

    data_format = K.image_data_format()
    if data_format == 'channels_first':
        channel_index = 1
    elif data_format == 'channels_last':
        channel_index = 3
    else:
        raise ValueError('Unknown data_format')
    inputs = Input(img_shape)

    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu',
                    name='conv_1', kernel_regularizer=l2(l2_reg))(inputs)
    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    # conv_2 = merge([
    #     Conv2D(128, (5, 5), activation="relu", name='conv_2_'+str(i+1),
    #            kernel_regularizer=l2(l2_reg))(
    #         splittensor(axis=channel_index, ratio_split=2, id_split=i)(conv_2)
    #     ) for i in range(2)], mode='concat', concat_axis=channel_index, name="conv_2")

    conv_2 = layers.concatenate([
        Conv2D(128, (5, 5), activation="relu", name='conv_2_'+str(i+1),
               kernel_regularizer=l2(l2_reg))(
            splittensor(axis=channel_index, ratio_split=2, id_split=i)(conv_2)
        ) for i in range(2)], axis=channel_index, name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_3',
                    kernel_regularizer=l2(l2_reg))(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = layers.concatenate([
        Conv2D(192, (3, 3), activation="relu", name='conv_4_'+str(i+1),
               kernel_regularizer=l2(l2_reg))(
            splittensor(axis=channel_index, ratio_split=2, id_split=i)(conv_4)
        ) for i in range(2)], axis=channel_index, name="conv_4")

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = layers.concatenate([
        Conv2D(128, (3, 3), activation="relu", name='conv_5_'+str(i+1),
               kernel_regularizer=l2(l2_reg))(
            splittensor(axis=channel_index, ratio_split=2, id_split=i)(conv_5)
        ) for i in range(2)], axis=channel_index, name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1',
                    kernel_regularizer=l2(l2_reg))(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2',
                    kernel_regularizer=l2(l2_reg))(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    if n_classes == 1000:
        dense_3 = Dense(n_classes, name='dense_3',
                        kernel_regularizer=l2(l2_reg))(dense_3)
    else:
        # We change the name so when loading the weights_file from a
        # Imagenet pretrained model does not crash
        dense_3 = Dense(n_classes, name='dense_3_new',
                        kernel_regularizer=l2(l2_reg))(dense_3)
    prediction = Activation("softmax", name="softmax")(dense_3)

    model = Model(inputs=inputs, outputs=prediction)

    return model
