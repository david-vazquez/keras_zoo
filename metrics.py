from __future__ import print_function

from keras import backend as K


def cce_flatt(void_class):
    def categorical_crossentropy_flatt(y_true, y_pred):
        '''Expects a binary class matrix instead of a vector of scalar classes.
        '''
        y_pred = K.permute_dimensions(y_pred, (0, 2, 3, 1))
        shp_y_pred = K.shape(y_pred)
        y_pred = K.reshape(y_pred, (shp_y_pred[0]*shp_y_pred[1]*shp_y_pred[2],
                           shp_y_pred[3]))  # go back to b01,c
        shp_y_true = K.shape(y_true)
        y_true = K.cast(K.flatten(y_true), 'int32')  # b,01 -> b01
        # remove void classes from cross_entropy
        if len(void_class):
            for i in range(len(void_class)):
                # get idx of non void classes and remove void classes
                # from y_true and y_pred
                idxs = K.not_equal(y_true, void_class[i]).nonzero()
                y_pred = y_pred[idxs]
                y_true = y_true[idxs]
        out = K.categorical_crossentropy(y_pred, y_true)

        return K.mean(out)  # b01 -> b,01
    return categorical_crossentropy_flatt
