from theano import tensor as T
from keras.utils.np_utils import conv_input_length


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


def _preprocess_conv2d_filter_shape(dim_ordering, filter_shape):
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


# NOTE: Only this funtion is changed from keras
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
    filter_shape = _preprocess_conv2d_filter_shape(dim_ordering, filter_shape)
    filter_shape = tuple(filter_shape[i] for i in (1, 0, 2, 3))

    op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(imshp=output_shape,
                                                        kshp=filter_shape,
                                                        subsample=strides,
                                                        border_mode=th_border_mode,
                                                        filter_flip=not flip_filters)

    # Set output size as [None, None]
    output_size = [None] * 2

    # Check if the width and heigth dimensions are None
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
