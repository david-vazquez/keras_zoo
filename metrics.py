from __future__ import print_function

from keras import backend as K


def categorical_crossentropy_flatt(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    shp_y_pred = K.shape(y_pred)
    y_pred = K.reshape(y_pred, (shp_y_pred[0]*shp_y_pred[1]*shp_y_pred[2],
                       shp_y_pred[3]))  # go back to b01,c
    shp_y_true = K.shape(y_true)
    y_true = K.cast(K.flatten(y_true), 'int32')  # b,01 -> b01
    out = K.categorical_crossentropy(y_pred, y_true)

    return K.mean(K.reshape(out, (shp_y_true[0],
                  shp_y_true[2]*shp_y_true[3])), axis=1)  # b01 -> b,01
