from keras.layers.convolutional import Convolution2D
from keras.utils.np_utils import conv_input_length
from keras import backend as K
dim_ordering = K.image_dim_ordering()
if dim_ordering == 'th':
    from deconv_th import deconv2d
else:
    from deconv_tf import deconv2d


class Deconvolution2D(Convolution2D):
    '''Transposed convolution operator for filtering windows of two-dimensional inputs.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.
    '''
    def __init__(self, nb_filter, nb_row, nb_col, input_shape,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1),
                 dim_ordering=K.image_dim_ordering(),
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for DeConv2D:', border_mode)

        self.output_shape_ = self.get_output_shape_for_helper(input_shape, nb_filter,
                                                              dim_ordering, nb_row, nb_col,
                                                              border_mode, subsample)
        super(Deconvolution2D, self).__init__(nb_filter, nb_row, nb_col,
                                              init=init, activation=activation,
                                              weights=weights, border_mode=border_mode,
                                              subsample=subsample, dim_ordering=dim_ordering,
                                              W_regularizer=W_regularizer, b_regularizer=b_regularizer,
                                              activity_regularizer=activity_regularizer,
                                              W_constraint=W_constraint, b_constraint=b_constraint,
                                              bias=bias, **kwargs)

    def get_output_shape_for_helper(self, input_shape,
                                    nb_filter, dim_ordering,
                                    nb_row, nb_col,
                                    border_mode, subsample):
        if dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)

        rows = conv_input_length(rows, nb_row,
                                 border_mode, subsample[0])
        cols = conv_input_length(cols, nb_col,
                                 border_mode, subsample[1])

        if dim_ordering == 'th':
            return (input_shape[0], nb_filter, rows, cols)
        elif dim_ordering == 'tf':
            return (input_shape[0], rows, cols, nb_filter)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            # rows = input_shape[1]
            # cols = input_shape[2]
            rows = self.output_shape_[1]
            cols = self.output_shape_[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        # rows = conv_input_length(rows, self.nb_row,
        #                          self.border_mode, self.subsample[0])
        # cols = conv_input_length(cols, self.nb_col,
        #                          self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)

    def call(self, x, mask=None):
        output = deconv2d(x, self.W, self.output_shape_,
                          strides=self.subsample,
                          border_mode=self.border_mode,
                          dim_ordering=self.dim_ordering,
                          filter_shape=self.W_shape)
        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = self.activation(output)
        return output
