from keras import backend as K
dim_ordering = K.image_dim_ordering()
if dim_ordering == 'th':
    from theano import tensor as T

from keras.layers.convolutional import Convolution2D
from keras.utils.np_utils import conv_output_length, conv_input_length
def _preprocess_conv2d_input(x, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        x = x.dimshuffle((0, 3, 1, 2))
    return x


def _preprocess_conv2d_kernel(kernel, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        kernel = kernel.dimshuffle((3, 2, 0, 1))
    return kernel


def _preprocess_border_mode(border_mode):
    if border_mode == 'same':
        th_border_mode = 'half'
    elif border_mode == 'valid':
        th_border_mode = 'valid'
    else:
        raise Exception('Border mode not supported: ' + str(border_mode))
    return th_border_mode


def _preprocess_image_shape(dim_ordering, image_shape):
    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None
    if dim_ordering == 'tf':
        if image_shape:
            image_shape = (image_shape[0], image_shape[3],
                           image_shape[1], image_shape[2])
    if image_shape is not None:
        image_shape = tuple(int_or_none(v) for v in image_shape)
    return image_shape


def _preprocess_filter_shape(dim_ordering, filter_shape):
    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None
    if dim_ordering == 'tf':
        if filter_shape:
            filter_shape = (filter_shape[3], filter_shape[2],
                            filter_shape[0], filter_shape[1])
    if filter_shape is not None:
        filter_shape = tuple(int_or_none(v) for v in filter_shape)
    return filter_shape


def _postprocess_conv2d_output(conv_out, x, border_mode, np_kernel, strides, dim_ordering):
    if border_mode == 'same':
        if np_kernel.shape[2] % 2 == 0:
            conv_out = conv_out[:, :, :(x.shape[2] + strides[0] - 1) // strides[0], :]
        if np_kernel.shape[3] % 2 == 0:
            conv_out = conv_out[:, :, :, :(x.shape[3] + strides[1] - 1) // strides[1]]
    if dim_ordering == 'tf':
        conv_out = conv_out.dimshuffle((0, 2, 3, 1))
    return conv_out

def deconv2d(x, kernel, output_shape, strides=(1, 1),
             border_mode='valid',
             dim_ordering='th',
             image_shape=None, filter_shape=None):
    '''2D deconvolution (transposed convolution).
    # Arguments
        kernel: kernel tensor.
        output_shape: desired dimensions of output.
        strides: strides tuple.
        border_mode: string, "same" or "valid".
        dim_ordering: "tf" or "th".
            Whether to use Theano or TensorFlow dimension ordering
        in inputs/kernels/ouputs.
    '''
    flip_filters = False
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv2d_input(x, dim_ordering)
    kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
    kernel = kernel.dimshuffle((1, 0, 2, 3))
    th_border_mode = _preprocess_border_mode(border_mode)
    np_kernel = kernel.eval()
    filter_shape = _preprocess_filter_shape(dim_ordering, filter_shape)

    op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(imshp=output_shape,
                                                        kshp=filter_shape,
                                                        subsample=strides,
                                                        border_mode=th_border_mode,
                                                        filter_flip=not flip_filters)

    output_size = [None] * 2


    for i in [-2, -1]:
        if output_shape[i] is None:
            output_size[i] = conv_input_length(x.shape[i], filter_shape[i],
                                                border_mode, strides[i])
        else:
            output_size[i] = output_shape[i]

    conv_out = op(kernel, x, output_size)
    conv_out = _postprocess_conv2d_output(conv_out, x, border_mode, np_kernel,
                                          strides, dim_ordering)
    return conv_out





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
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)


    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_input_length(rows, self.nb_row,
                                 self.border_mode, self.subsample[0])
        cols = conv_input_length(cols, self.nb_col,
                                 self.border_mode, self.subsample[1])
        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        # out_shape_tmp = self.output_shape_[:2] + (1, 1)
        # print(out_shape_tmp)
        out_shape_tmp = self.output_shape_
        output = deconv2d(x, self.W, out_shape_tmp,
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
