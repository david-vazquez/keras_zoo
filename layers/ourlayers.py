from keras import backend as K
dim_ordering = K.image_dim_ordering()
if dim_ordering == 'th':
    import theano
    from theano import tensor as T
    from theano.scalar.basic import Inv

from keras import backend as K
from keras.layers.core import Layer
from keras.layers import UpSampling2D


# Function from Lasagne framework
def get_input_shape(output_length, filter_size, stride, pad=0):
    """Helper function to compute the input size of a convolution operation
    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.
    Parameters
    ----------
    output_length : int or None
        The size of the output.
    filter_size : int
        The size of the filter.
    stride : int
        The stride of the convolution operation.
    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.
        A single integer results in symmetric zero-padding of the given size on
        both borders.
        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.
        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.
    Returns
    -------
    int or None
        The smallest input size corresponding to the given convolution
        parameters for the given output size, or ``None`` if `output_size` is
        ``None``. For a strided convolution, any input size of up to
        ``stride - 1`` elements larger than returned will still give the same
        output size.
    Raises
    ------
    ValueError
        When an invalid padding is specified, a `ValueError` is raised.
    Notes
    -----
    This can be used to compute the output size of a convolution backward pass,
    also called transposed convolution, fractionally-strided convolution or
    (wrongly) deconvolution in the literature.
    """
    if output_length is None:
        return None
    if pad == 'valid':
        pad = 0
    elif pad == 'full':
        pad = filter_size - 1
    elif pad == 'same':
        pad = filter_size // 2
    if not isinstance(pad, int):
        raise ValueError('Invalid pad: {0}'.format(pad))
    return (output_length - 1) * stride - 2 * pad + filter_size


# Works TH and TF
class CropLayer2D(Layer):
    def __init__(self, img_in, *args, **kwargs):
        self.img_in = img_in
        self.dim_ordering = K.image_dim_ordering()
        super(CropLayer2D, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.crop_size = self.img_in._keras_shape[-2:]
        if self.dim_ordering == 'tf':
            self.crop_size = self.img_in._keras_shape[1:3]
        super(CropLayer2D, self).build(input_shape)

    def call(self, x, mask=False):
        input_shape = K.shape(x)
        cs = K.shape(self.img_in)
        if self.dim_ordering == 'th':
            input_shape = input_shape[-2:]
            cs = cs[-2:]
        else:
            input_shape = input_shape[1:3]
            cs = cs[1:3]
        dif = (input_shape - cs)/2
        if self.dim_ordering == 'th':
            if K.ndim(x) == 5:
                return x[:, :, :, dif[0]:dif[0]+cs[0], dif[1]:dif[1]+cs[1]]
            return x[:, :, dif[0]:dif[0]+cs[0], dif[1]:dif[1]+cs[1]]
        else:
            if K.ndim(x) == 5:
                return x[:, :, dif[0]:dif[0]+cs[0], dif[1]:dif[1]+cs[1], :]
            return x[:, dif[0]:dif[0]+cs[0], dif[1]:dif[1]+cs[1], :]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            return tuple(input_shape[:-2]) + (self.crop_size[0],
                                              self.crop_size[1])
        if self.dim_ordering == 'tf':
            return ((input_shape[:1], ) + tuple(self.crop_size) +
                    (input_shape[-1], ))

    # def get_config(self):
    #     config = {'img_in': self.img_in,
    #               'dim_ordering': self.dim_ordering}
    #     base_config = super(CropLayer2D, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))


class MergeSequences(Layer):
    def __init__(self, merge=True, batch_size=None, *args, **kwargs):
        self.merge = merge
        self.batch_size = batch_size
        if K._BACKEND != "theano":
            raise NotImplementedError("Check the unbroadcast in TensorFlow")

        super(MergeSequences, self).__init__(*args, **kwargs)

    def get_output_shape_for(self, input_shape):
        sh = input_shape
        bs = self.batch_size
        if self.merge:
            if sh[0] is None or sh[1] is None:
                return (None, ) + tuple(sh[2:])
            return [[sh[0]*sh[1]] + list(sh[2:])]
        else:
            if sh[0] is None:
                return (bs, None, ) + tuple(sh[1:])
            # keras bug keras/engine/training.py", line 104, in
            # standardize_input_data str(array.shape))
            # return tuple([bs, sh[0]/bs] + list(sh[1:]))
            return tuple([bs, sh[0]/bs]) + (sh[1], None, None)

    def call(self, x, mask=None):
        sh = x.shape
        bs = self.batch_size
        if self.merge:
            sh = (sh[0]*sh[1], ) + tuple(sh[2:])
            return T.reshape(x, sh, ndim=4)
        else:
            sh = (bs, sh[0]/bs, ) + tuple(sh[1:])
            ret = T.reshape(x, sh, ndim=5)
            return T.unbroadcast(ret, 0)


# Works TH and TF
class NdSoftmax(Layer):
    '''N-dimensional Softmax
    Will compute the Softmax on channel_idx and return a tensor of the
    same shape as the input
    '''
    def __init__(self, dim_ordering='default', *args, **kwargs):

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if dim_ordering == 'th':
            self.channel_index = 1
        if dim_ordering == 'tf':
            self.channel_index = 3

        super(NdSoftmax, self).__init__(*args, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        ch_idx = self.channel_index
        l_idx = K.ndim(x) - 1  # last index
        x = K.permute_dimensions(
            x, tuple(i for i in range(K.ndim(x)) if i != ch_idx) + (ch_idx,))
        sh = K.shape(x)
        x = K.reshape(x, (-1, sh[-1]))
        x = K.softmax(x)
        x = K.reshape(x, sh)
        x = K.permute_dimensions(
            x, tuple(range(ch_idx) + [l_idx] + range(ch_idx, l_idx)))
        return x


# Works TH and TF
class DePool2D(UpSampling2D):
    '''Simplar to UpSample, yet traverse only maxpooled elements
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        4D tensor with shape:
        `(samples, channels, upsampled_rows, upsampled_cols)` if
        dim_ordering='th' or 4D tensor with shape:
        `(samples, upsampled_rows, upsampled_cols, channels)` if
        dim_ordering='tf'.
    # Arguments
        size: tuple of 2 integers. The upsampling factors for rows and columns.
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 4

    def __init__(self, pool2d_layer, *args, **kwargs):
        self._pool2d_layer = pool2d_layer
        super(DePool2D, self).__init__(*args, **kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.dim_ordering == 'th':
            output = K.repeat_elements(X, self.size[0], axis=2)
            output = K.repeat_elements(output, self.size[1], axis=3)
        elif self.dim_ordering == 'tf':
            output = K.repeat_elements(X, self.size[0], axis=1)
            output = K.repeat_elements(output, self.size[1], axis=2)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        f = T.grad(T.sum(self._pool2d_layer.get_output(train)),
                   wrt=self._pool2d_layer.get_input(train)) * output

        return f


# 1D Bilinear interpolation for even size filters
def bilinear1D(ratio, normalize=True):
    half_kern = T.arange(1, ratio+1, dtype=theano.config.floatX)
    kern = T.concatenate([half_kern, half_kern[-1::-1]])
    if normalize:
        kern /= (ratio)
    return kern


# 2D Bilinear interpolation for even size filters
def bilinear2D(ratio, normalize=True):
    hkern = bilinear1D(ratio=ratio, normalize=normalize).dimshuffle('x', 0)
    vkern = bilinear1D(ratio=ratio, normalize=normalize).dimshuffle(0, 'x')
    kern = hkern * vkern
    return kern


# 4D Bilinear interpolation for even size filters
def bilinear4D_(ratio, num_input_channels, num_filters, normalize=True):
    kern = bilinear2D(ratio=ratio, normalize=normalize)
    kern = kern.dimshuffle('x', 'x', 0, 1)
    kern = T.extra_ops.repeat(kern, num_input_channels, axis=0)
    kern = T.extra_ops.repeat(kern, num_filters, axis=1)
    return kern.eval()


def bilinear4D(ratio, num_input_channels, num_filters, normalize=True):
    W = bilinear4D_(ratio, num_input_channels, num_filters, normalize)
    for i in range(num_input_channels):
        for j in range(num_filters):
            if i != j:
                W[i, j, :, :] = 0
    return W


# 4D Bilinear interpolation for even size filters
def bilinear4D_T(ratio, num_input_channels, num_filters, normalize=True):
    kern = T.nnet.abstract_conv.bilinear_kernel_2D(ratio, normalize=True)
    kern = kern.dimshuffle('x', 'x', 0, 1)
    kern = T.extra_ops.repeat(kern, num_input_channels, axis=0)
    kern = T.extra_ops.repeat(kern, num_filters, axis=1)
    return kern.eval()
