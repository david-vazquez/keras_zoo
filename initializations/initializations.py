import numpy as np
from keras import backend as K

# Create a 2D bilinear interpolation kernel in numpy for even size filters
def bilinear(w, h):
    import math
    data = np.zeros((w*h), dtype=float)
    f = math.ceil(w / 2.)
    c = (2 * f - 1 - f % 2) / (2. * f)
    # print ('f:{}, c:{}'.format(f, c))
    for i in range(w*h):
        x = float(i % w)
        y = float((i / w) % h)
        v = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        # print ('x:{}, y:{}, v:{}'.format(x, y, v))
        np.put(data, i, v)
    data = data.reshape((h, w))
    return data


# Create 4D bilinear interpolation kernel in numpy for even size filters
def bilinear4D(w, h, num_input_channels, num_filters):
    kern = bilinear(w, h)
    kern = kern.reshape((1, 1, w, h))
    kern = np.repeat(kern, num_input_channels, axis=0)
    kern = np.repeat(kern, num_filters, axis=1)
    for i in range(num_input_channels):
        for j in range(num_filters):
            if i != j:
                kern[i, j, :, :] = 0
    return kern


# Create a Keras bilinear weight initializer
def bilinear_init(shape, name=None, dim_ordering='th'):
    # print ('Shape: '),
    # print (shape)
    kernel = bilinear4D(shape[0], shape[1], shape[2], shape[3])
    np.set_printoptions(threshold=np.nan)
    kernel = kernel.transpose((2, 3, 0, 1))
    # print (kernel)
    # print (kernel.shape)
    kvar = K.variable(value=kernel, dtype=K.floatx(), name='bilinear')
    return kvar
