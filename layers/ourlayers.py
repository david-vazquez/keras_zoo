from theano import tensor as T

from keras import backend as K
from keras.layers.core import Layer


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


class CropLayer2D(Layer):
    def __init__(self, crop_shape, dim_ordering='th', *args, **kwargs):
        self.crop_shape = crop_shape
        assert dim_ordering in ['tf', 'th'], ('dim_ordering must be '
                                              'in [tf, th]')
        self.dim_ordering = dim_ordering

        super(CropLayer2D, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.crop_size = self.crop_shape[-2:]
            # self.diff = input_shape[-2:] - self.crop_size
        if self.dim_ordering == 'tf':
            self.crop_size = self.crop_shape[1:3]
            # self.diff = input_shape[-3:-1] - self.crop_size

        super(CropLayer2D, self).build(input_shape)

    def call(self, x, mask=False):
        input_shape = x.shape
        cs = self.crop_size
        if self.dim_ordering == 'th':
            dif = input_shape[-2:] - cs
        if self.dim_ordering == 'tf':
            dif = input_shape[-3:-1] - cs
        dif = dif/2
        if self.dim_ordering == 'th':
            if K.ndim(x) == 5:
                return x[:, :, :, dif[0]:dif[0]+cs[0], dif[1]:dif[1]+cs[1]]
            return x[:, :, dif[0]:dif[0]+cs[0], dif[1]:dif[1]+cs[1]]
        if self.dim_ordering == 'tf':
            if K.ndim(x) == 5:
                return x[:, :, dif[0]:dif[0]+cs[0], dif[1]:dif[1]+cs[1], :]
            return x[:, dif[0]:dif[0]+cs[0], dif[1]:dif[1]+cs[1], :]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            # if self.ndim == 5:
            #     return (input_shape[0], input_shape[1], input_shape[2],
            #             self.crop_size[0], self.crop_size[1]))
            # else:
            return tuple(input_shape[:-2]) + (self.crop_size[0],
                                              self.crop_size[1])
        if self.dim_ordering == 'tf':
            return ((input_shape[:1], ) + tuple(self.crop_size) +
                    (input_shape[-1], ))


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


class NdSoftmax(Layer):
    '''N-dimensional Softmax
    Will compute the Softmax on channel_idx and return a tensor of the
    same shape as the input
    '''
    def __init__(self, channel_idx, *args, **kwargs):
        self.channel_idx = channel_idx
        super(NdSoftmax, self).__init__(*args, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        ch_idx = self.channel_idx
        l_idx = x.ndim - 1  # last index
        x = K.permute_dimensions(
            x, tuple(i for i in range(x.ndim) if i != ch_idx) + (ch_idx,))
        sh = K.shape(x)
        x = K.reshape(x, (-1, sh[-1]))
        x = K.softmax(x)
        x = K.reshape(x, sh)
        x = K.permute_dimensions(
            x, tuple(range(ch_idx) + [l_idx] + range(ch_idx, l_idx)))
        return x
