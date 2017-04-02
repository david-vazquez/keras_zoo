from keras import backend as K
import tensorflow as tf
from keras.utils.np_utils import conv_input_length


def _preprocess_conv2d_input(x, dim_ordering):
    if K.dtype(x) == 'float64':
        x = tf.cast(x, 'float32')
    if dim_ordering == 'th':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        x = tf.transpose(x, (0, 2, 3, 1))
    return x


def _preprocess_deconv_output_shape(x, shape, dim_ordering):
    if dim_ordering == 'th':
        shape = (shape[0], shape[2], shape[3], shape[1])

    if shape[0] is None:
        shape = (tf.shape(x)[0], ) + tuple(shape[1:])
        shape = tf.stack(list(shape))
    return shape


def _preprocess_conv2d_kernel(kernel, dim_ordering):
    if K.dtype(kernel) == 'float64':
        kernel = tf.cast(kernel, 'float32')
    if dim_ordering == 'th':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        kernel = tf.transpose(kernel, (2, 3, 1, 0))
    return kernel


def _preprocess_border_mode(border_mode):
    if border_mode == 'same':
        padding = 'SAME'
    elif border_mode == 'valid':
        padding = 'VALID'
    else:
        raise ValueError('Invalid border mode:', border_mode)
    return padding


def _postprocess_conv2d_output(x, dim_ordering):
    if dim_ordering == 'th':
        x = tf.transpose(x, (0, 3, 1, 2))

    if K.floatx() == 'float64':
        x = tf.cast(x, 'float64')
    return x


# NOTE: Only this funtion is changed from keras
def deconv2d(x, kernel, output_shape, strides=(1, 1),
             border_mode='valid',
             dim_ordering='default',
             image_shape=None, filter_shape=None):
    """2D deconvolution (i.e. transposed convolution).
    # Arguments
        x: input tensor.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        border_mode: string, `"same"` or `"valid"`.
        dim_ordering: `"tf"` or `"th"`.
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.
    # Returns
        A tensor, result of transposed 2D convolution.
    # Raises
        ValueError: if `dim_ordering` is neither `tf` or `th`.
    """
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv2d_input(x, dim_ordering)
    strides = (1,) + strides + (1,)

    # Get output shape
    shape_b = tf.shape(x)[0]
    shape_h = output_shape[1]
    shape_w = output_shape[2]
    shape_c = output_shape[3]

    # print('Output Shape: ' + str(output_shape))
    # print('Input Shape: ' + str(x.get_shape()))

    # Compute output height if none
    if shape_h is None:
        shape_h = conv_input_length(tf.shape(x)[1], filter_shape[0],
                                    border_mode, strides[1])

    # Compute output width if none
    if shape_w is None:
        shape_w = conv_input_length(tf.shape(x)[2], filter_shape[1],
                                    border_mode, strides[2])

    # Compose output shape without nones
    try:
        # Uses tf.pack, previous to tensorflow 1.0.0
        output_shape = tf.pack([shape_b, shape_h, shape_w, shape_c])
    except AttributeError:
        # Uses tf.stack in favor of tf.pack, which is deprecated from tensorflow v1.0.0 onwards
        output_shape = tf.stack([shape_b, shape_h, shape_w, shape_c])

    output_shape = _preprocess_deconv_output_shape(x, output_shape,
                                                   dim_ordering)
    kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
    kernel = tf.transpose(kernel, (0, 1, 3, 2))
    padding = _preprocess_border_mode(border_mode)

    x = tf.nn.conv2d_transpose(x, kernel, output_shape, strides,
                               padding=padding)
    return _postprocess_conv2d_output(x, dim_ordering)
