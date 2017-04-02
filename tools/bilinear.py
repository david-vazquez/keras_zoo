#! /usr/bin/env python
from numpy import *
import argparse

def bilinear_kernel(ratio, normalize = False):
    '''
    For upsample ratio of ratio,
    Kernel size is ratio*2-1
    '''
    assert ratio > 1
    kern_size = ratio * 2 - 1
    kern = zeros((kern_size, kern_size))
    for ii in range(-ratio+1, ratio):
        for jj in range(-ratio+1, ratio):
            kern[ii+ratio-1,jj+ratio-1] = (ratio-abs(ii)) * (ratio-abs(jj))
    if normalize:
        kern /= ratio**2
    return kern

def bilinear_weights(dim, ratio, normalize = True):
    '''
    creates blinear tensor for convolution.
    :type dim: int
    :param dim: the number of channels to be upsampled
    :type dim: ratio
    :param dim: the upsampling ratio
    :type normalize: bool
    : param normalize: whether to normalize the kernel or not
    returns: a 4D tensor of shape (dim, dim, bilinear_kernel_row, bilinear_kernel_col)
             where all elements are zero, except elements in the diagonal of the first
             two dimensions, e.g. [0,0,:,:], [1,1,:,:], [2,2,:,:]
    '''
    kernel = bilinear_kernel(ratio, normalize=normalize)
    row, col = kernel.shape
    bl_weights = zeros((dim, dim, row, col))
    bl_weights[range(dim),range(dim)] = kernel
    return bl_weights

def main():
    parser = argparse.ArgumentParser(description='Computes bilinear kernel')
    parser.add_argument('upsample_ratio', type = int, default = 3, help = 'ratio to upsample by')
    args = parser.parse_args()

    print 'Kernel for an upsample ratio of %d is\n1/%d *' % (args.upsample_ratio, args.upsample_ratio**2)
    print bilinear_kernel(args.upsample_ratio)
    print '='
    print bilinear_kernel(args.upsample_ratio, normalize = True)



if __name__ == '__main__':
    main()
