from __future__ import absolute_import
from __future__ import print_function

import warnings
import skimage.io as io
from skimage.color import rgb2gray, gray2rgb
import skimage.transform
import numpy as np
from numpy import ma
from numpy.linalg import inv
from six.moves import range
import os
import SimpleITK as sitk

from keras import backend as K
from keras.preprocessing.image import (Iterator,
                                       img_to_array,
                                       transform_matrix_offset_center,
                                       apply_transform,
                                       flip_axis,
                                       array_to_img,
                                       NumpyArrayIterator,
                                       random_channel_shift)

from tools.save_images import save_img2
from tools.yolo_utils import yolo_build_gt_batch

# Pad image
def pad_image(x, pad_amount, mode='reflect', constant=0.):
    e = pad_amount
    shape = list(x.shape)
    shape[:2] += 2*e
    if mode == 'constant':
        x_padded = np.ones(shape, dtype=np.float32)*constant
        x_padded[e:-e, e:-e] = x.copy()
    else:
        x_padded = np.zeros(shape, dtype=np.float32)
        x_padded[e:-e, e:-e] = x.copy()

    if mode == 'reflect':
        x_padded[:e, e:-e] = np.flipud(x[:e, :])  # left edge
        x_padded[-e:, e:-e] = np.flipud(x[-e:, :])  # right edge
        x_padded[e:-e, :e] = np.fliplr(x[:, :e])  # top edge
        x_padded[e:-e, -e:] = np.fliplr(x[:, -e:])  # bottom edge
        x_padded[:e, :e] = np.fliplr(np.flipud(x[:e, :e]))  # top-left corner
        x_padded[-e:, :e] = np.fliplr(np.flipud(x[-e:, :e]))  # top-right cor
        x_padded[:e, -e:] = np.fliplr(np.flipud(x[:e, -e:]))  # bot-left cor
        x_padded[-e:, -e:] = np.fliplr(np.flipud(x[-e:, -e:]))  # bot-right cor
    elif mode == 'zero' or mode == 'constant':
        pass
    elif mode == 'nearest':
        x_padded[:e, e:-e] = x[[0], :]  # left edge
        x_padded[-e:, e:-e] = x[[-1], :]  # right edge
        x_padded[e:-e, :e] = x[:, [0]]  # top edge
        x_padded[e:-e, -e:] = x[:, [-1]]  # bottom edge
        x_padded[:e, :e] = x[[0], [0]]  # top-left corner
        x_padded[-e:, :e] = x[[-1], [0]]  # top-right corner
        x_padded[:e, -e:] = x[[0], [-1]]  # bottom-left corner
        x_padded[-e:, -e:] = x[[-1], [-1]]  # bottom-right corner
    else:
        raise ValueError("Unsupported padding mode \"{}\"".format(mode))
    return x_padded


# Define warp
def gen_warp_field(shape, sigma=0.1, grid_size=3):
    # Initialize bspline transform
    args = shape+(sitk.sitkFloat32,)
    ref_image = sitk.Image(*args)
    tx = sitk.BSplineTransformInitializer(ref_image, [grid_size, grid_size])

    # Initialize shift in control points:
    # mesh size = number of control points - spline order
    p = sigma * np.random.randn(grid_size+3, grid_size+3, 2)

    # Anchor the edges of the image
    p[:, 0, :] = 0
    p[:, -1:, :] = 0
    p[0, :, :] = 0
    p[-1:, :, :] = 0

    # Set bspline transform parameters to the above shifts
    tx.SetParameters(p.flatten())

    # Compute deformation field
    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(ref_image)
    displacement_field = displacement_filter.Execute(tx)

    return displacement_field


# Apply warp
def apply_warp(x, warp_field, fill_mode='reflect',
               interpolator=sitk.sitkLinear,
               fill_constant=0):
    # Expand deformation field (and later the image), padding for the largest
    # deformation
    warp_field_arr = sitk.GetArrayFromImage(warp_field)
    max_deformation = np.max(np.abs(warp_field_arr))
    pad = np.ceil(max_deformation).astype(np.int32)
    warp_field_padded_arr = pad_image(warp_field_arr, pad_amount=pad,
                                      mode='nearest')
    warp_field_padded = sitk.GetImageFromArray(warp_field_padded_arr,
                                               isVector=True)

    # Warp x, one filter slice at a time
    x_warped = np.zeros(x.shape, dtype=np.float32)
    warp_filter = sitk.WarpImageFilter()
    warp_filter.SetInterpolator(interpolator)
    warp_filter.SetEdgePaddingValue(np.min(x).astype(np.double))
    for i, image in enumerate(x):
        image_padded = pad_image(image, pad_amount=pad, mode=fill_mode,
                                 constant=fill_constant).T
        image_f = sitk.GetImageFromArray(image_padded)
        image_f_warped = warp_filter.Execute(image_f, warp_field_padded)
        image_warped = sitk.GetArrayFromImage(image_f_warped)
        x_warped[i] = image_warped[pad:-pad, pad:-pad].T

    return x_warped


# List the subdirectories in a directory
def list_subdirs(directory):
    subdirs = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            subdirs.append(subdir)
    return subdirs


# Checks if a file is an image
def has_valid_extension(fname, white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'tif'}):
    for extension in white_list_formats:
        if fname.lower().endswith('.' + extension):
            return True
    return False


# Load image
def load_img(path, grayscale=False, resize=None, order=1):
    # Load image
    img = io.imread(path)

    # Resize
    # print('Desired resize: ' + str(resize))
    if resize is not None:
        img = skimage.transform.resize(img, resize, order=order,
                                       preserve_range=True)
        # print('Final resize: ' + str(img.shape))

    # Color conversion
    if len(img.shape)==2 and not grayscale:
        img = gray2rgb(img)
    elif len(img.shape)>2 and img.shape[2]==3 and grayscale:
        img = rgb2gray(img)

    # Return image
    return img


class ImageDataGenerator(object):
    '''Generate minibatches withGT4_DAComb_3cl_224x224_rescale_lr10-4_noCWB
    real-time data augmentation.
    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before
            applying any other transformation).
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
    '''
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 gcn=False,#
                 imageNet=False,#
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 void_label=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 spline_warp=False,
                 warp_sigma=0.1,
                 warp_grid_size=3,
                 dim_ordering='default',
                 class_mode='categorical',
                 rgb_mean=None,
                 rgb_std=None,
                 crop_size=None):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.__dict__.update(locals())
        self.principal_components = None
        # self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.cb_weights = None

        if dim_ordering not in {'tf', 'th'}:
            raise Exception('dim_ordering should be "tf" (channel after row '
                            'and column) or "th" (channel before row and '
                            'column). Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        # Broadcast the shape of mean and std
        if rgb_mean is not None and featurewise_center:
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_index - 1] = len(rgb_mean)
            self.mean = np.reshape(rgb_mean, broadcast_shape)
            print ('   Mean {}: {}'.format(self.mean.shape, self.rgb_mean))

        # Broadcast the shape of std
        if rgb_std is not None and featurewise_std_normalization:
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_index - 1] = len(rgb_std)
            self.std = np.reshape(rgb_std, broadcast_shape)
            print ('   Std {}: {}'.format(self.std.shape, self.rgb_std))

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)

        # Check class mode
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'segmentation', 'detection', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "segmentation", "detection" or None.')
        self.class_mode = class_mode
        self.has_gt_image = True if self.class_mode == 'segmentation' else False

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_format=save_format)

    def flow_from_directory(self, directory,
                            resize=None, target_size=(256, 256),
                            color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            gt_directory=None,
                            save_to_dir=None, save_prefix='',
                            save_format='jpeg'):
        return DirectoryIterator(
            directory, self, resize=resize,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            gt_directory=gt_directory,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_format=save_format)

    def flow_from_directory2(self, directory,
                             resize=None, target_size=(256, 256),
                             color_mode='rgb',
                             classes=None, class_mode='categorical',
                             batch_size=32, shuffle=True, seed=None,
                             gt_directory=None,
                             save_to_dir=None, save_prefix='',
                             save_format='jpeg', directory2=None,
                             gt_directory2=None, batch_size2=None):
        return DirectoryIterator2(
            directory, self, resize=resize,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            gt_directory=gt_directory,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_format=save_format,
            directory2=directory2, gt_directory2=gt_directory2,
            batch_size2=batch_size2)

    def standardize(self, x, y=None):
        if self.imageNet:
            if self.dim_ordering == 'th':
                # 'RGB'->'BGR'
                x = x[::-1, :, :]
                # Zero-center by mean pixel
                x[0, :, :] -= 103.939
                x[1, :, :] -= 116.779
                x[2, :, :] -= 123.68
            else:
                # 'RGB'->'BGR'
                x = x[:, :, ::-1]
                # Zero-center by mean pixel
                x[:, :, 0] -= 103.939
                x[:, :, 1] -= 116.779
                x[:, :, 2] -= 123.68
            return x

        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1

        if self.rescale:
            x *= self.rescale

        if self.gcn:
            x_before = x.copy()
            # Compute the void mask
            mask = np.ones_like(y).astype('int32')
            mask[y == self.void_label] = 0.
            mask = np.repeat(mask, 3, axis=0)

            # Mask image
            x_masked = ma.masked_array(x, mask=~mask.astype(bool))

            # Compute mean and std masked
            mean_masked = np.mean(x_masked)
            std_masked = np.std(x_masked)

            # Normalize
            s = 1
            eps = 1e-8
            x = s * (x - mean_masked) / max(eps, std_masked)

            # Set void pixels to 0
            x = x*mask


        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')

        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (x.size))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')

        return x

    def random_transform(self, x, y=None):
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        # prepare the data if GT is detection
        if self.class_mode == 'detection':
            h, w = x.shape[img_row_index], x.shape[img_col_index]
            # convert relative coordinates x,y,w,h to absolute x1,y1,x2,y2
            b = np.copy(y[:,1:5])
            b[:,0] = y[:,1]*w - y[:,3]*w/2
            b[:,1] = y[:,2]*h - y[:,4]*h/2
            b[:,2] = y[:,1]*w + y[:,3]*w/2
            b[:,3] = y[:,2]*h + y[:,4]*h/2

        # use composition of homographies to generate final transform that
        # needs to be applied
        need_transform = False

        # Rotation
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range,
                                                    self.rotation_range)
            need_transform = True
        else:
            theta = 0

        # Shift in height
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range,
                                   self.height_shift_range) * x.shape[img_row_index]
            need_transform = True
        else:
            tx = 0

        # Shift in width
        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range,
                                   self.width_shift_range) * x.shape[img_col_index]
            need_transform = True
        else:
            ty = 0

        # Shear
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
            need_transform = True
        else:
            shear = 0

        # Zoom
        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0],
                                       self.zoom_range[1], 2)
            need_transform = True


        if need_transform:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])

            translation_matrix = np.array([[1, 0, tx],
                                           [0, 1, ty],
                                           [0, 0, 1]])

            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])

            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])

            transform_matrix = np.dot(np.dot(np.dot(rotation_matrix,
                                                    translation_matrix),
                                             shear_matrix), zoom_matrix)

            h, w = x.shape[img_row_index], x.shape[img_col_index]
            transform_matrix = transform_matrix_offset_center(transform_matrix,
                                                              h, w)
            x = apply_transform(x, transform_matrix, img_channel_index,
                                fill_mode=self.fill_mode, cval=self.cval)
            if y is not None:
                if self.has_gt_image:
                    y = apply_transform(y, transform_matrix, img_channel_index,
                                        fill_mode=self.fill_mode, cval=self.void_label)
                elif self.class_mode == 'detection':
                    # point transformation is the inverse of image transformation
                    p_transform_matrix = inv(transform_matrix)
                    for ii in range(b.shape[0]):
                        x1,y1,x2,y2 = b.astype(int)[ii]
                        # get the four edge points of the bounding box
                        v1 = np.array([y1,x1,1])
                        v2 = np.array([y2,x2,1]) 
                        v3 = np.array([y2,x1,1])
                        v4 = np.array([y1,x2,1])
                        # transform the 4 points
                        v1 = np.dot(p_transform_matrix, v1)
                        v2 = np.dot(p_transform_matrix, v2)
                        v3 = np.dot(p_transform_matrix, v3)
                        v4 = np.dot(p_transform_matrix, v4)
                        # compute the new bounding box edges
                        b[ii,0] = np.min([v1[1],v2[1],v3[1],v4[1]]) 
                        b[ii,1] = np.min([v1[0],v2[0],v3[0],v4[0]])
                        b[ii,2] = np.max([v1[1],v2[1],v3[1],v4[1]])
                        b[ii,3] = np.max([v1[0],v2[0],v3[0],v4[0]]) 

        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range,
                                     img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                if y is not None:
                    if self.has_gt_image:
                        y = flip_axis(y, img_col_index)
                    elif self.class_mode == 'detection':
                        b[:,0],b[:,2] = w - b[:,2], w - b[:,0]

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                if y is not None:
                    if self.has_gt_image:
                        y = flip_axis(y, img_row_index)
                    elif self.class_mode == 'detection':
                        b[:,1],b[:,3] = h - b[:,3], h - b[:,1]

        if self.spline_warp:
            warp_field = gen_warp_field(shape=x.shape[-2:],
                                        sigma=self.warp_sigma,
                                        grid_size=self.warp_grid_size)
            x = apply_warp(x, warp_field,
                           interpolator=sitk.sitkLinear,
                           fill_mode=self.fill_mode, fill_constant=self.cval)

            if y is not None:
                if self.has_gt_image:
                    y = np.round(apply_warp(y, warp_field,
                                            interpolator=sitk.sitkNearestNeighbor,
                                            fill_mode=self.fill_mode,
                                            fill_constant=self.void_label))
                elif self.class_mode == 'detection':
                    raise ValueError('Elastic deformation is not supported for class_mode:', self.class_mode)

        # Crop
        # TODO: tf compatible???
        crop = list(self.crop_size) if self.crop_size else None
        if crop:
            # print ('X before: ' + str(x.shape))
            # print ('Y before: ' + str(y.shape))
            # print ('Crop_size: ' + str(self.crop_size))
            h, w = x.shape[img_row_index], x.shape[img_col_index]

            # Padd image if it is smaller than the crop size
            pad_h1, pad_h2, pad_w1, pad_w2 = 0, 0, 0, 0
            if h < crop[0]:
                total_pad = crop[0] - h
                pad_h1 = total_pad/2
                pad_h2 = total_pad-pad_h1
            if w < crop[1]:
                total_pad = crop[1] - w
                pad_w1 = total_pad/2
                pad_w2 = total_pad - pad_w1
            if h < crop[0] or w < crop[1]:
                x = np.lib.pad(x, ((0, 0), (pad_h1, pad_h2), (pad_w1, pad_w2)),
                               'constant')
                if y is not None:
                    if self.has_gt_image:
                        y = np.lib.pad(y, ((0, 0), (pad_h1, pad_h2), (pad_w1, pad_w2)),
                                       'constant', constant_values=self.void_label)
                    elif self.class_mode == 'detection':
                        b[:,0] = b[:,0] + pad_w1
                        b[:,1] = b[:,1] + pad_h1
                        b[:,2] = b[:,2] + pad_w1
                        b[:,3] = b[:,3] + pad_h1


                h, w = x.shape[img_row_index], x.shape[img_col_index]
                # print ('New size X: ' + str(x.shape))
                # print ('New size Y: ' + str(y.shape))
                # exit()

            if crop[0] < h:
                top = np.random.randint(h - crop[0])
            else:
                #print('Data augmentation: Crop height >= image size')
                top, crop[0] = 0, h
            if crop[1] < w:
                left = np.random.randint(w - crop[1])
            else:
                #print('Data augmentation: Crop width >= image size')
                left, crop[1] = 0, w

            if self.dim_ordering == 'th':
                x = x[..., :, top:top+crop[0], left:left+crop[1]]
                if y is not None:
                    if self.has_gt_image:
                        y = y[..., :, top:top+crop[0], left:left+crop[1]]
            else:
                x = x[..., top:top+crop[0], left:left+crop[1], :]
                if y is not None:
                    if self.has_gt_image:
                        y = y[..., top:top+crop[0], left:left+crop[1], :]

            if self.class_mode == 'detection':
                b[:,0] = b[:,0] - left
                b[:,1] = b[:,1] - top
                b[:,2] = b[:,2] - left
                b[:,3] = b[:,3] - top

            # print ('X after: ' + str(x.shape))
            # print ('Y after: ' + str(y.shape))

        if self.class_mode == 'detection':
            # clamp to valid coordinate values
            b[:,0] = np.clip( b[:,0] , 0 , w )
            b[:,1] = np.clip( b[:,1] , 0 , h )
            b[:,2] = np.clip( b[:,2] , 0 , w )
            b[:,3] = np.clip( b[:,3] , 0 , h )
            # convert back from absolute x1,y1,x2,y2 coordinates to relative x,y,w,h
            y[:,1] = (b[:,0] + (b[:,2]-b[:,0])/2 ) / w
            y[:,2] = (b[:,1] + (b[:,3]-b[:,1])/2 ) / h
            y[:,3] = (b[:,2]-b[:,0]) / w
            y[:,4] = (b[:,3]-b[:,1]) / h
            # reject regions that are too small
            y = y[y[:,3]>0.005]
            y = y[y[:,4]>0.005]

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        # hsv random shifts
        # blur
        return x, y

    def fit(self, X, augment=False, rounds=1, seed=None):
        """Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            X: Numpy array, the data to fit on. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        """
        X = np.asarray(X)
        if X.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(X.shape))
        if X.shape[self.channel_index] not in {1, 3, 4}:
            raise ValueError(
                'Expected input to be images (as Numpy array) '
                'following the dimension ordering convention "' + self.dim_ordering + '" '
                '(channels on axis ' + str(self.channel_index) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' + str(self.channel_index) + '. '
                'However, it was passed an array with shape ' + str(X.shape) +
                ' (' + str(X.shape[self.channel_index]) + ' channels).')

        if seed is not None:
            np.random.seed(seed)

        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=(0, self.row_index, self.col_index))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_index - 1] = X.shape[self.channel_index]
            self.mean = np.reshape(self.mean, broadcast_shape)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=(0, self.row_index, self.col_index))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_index - 1] = X.shape[self.channel_index]
            self.std = np.reshape(self.std, broadcast_shape)
            X /= (self.std + K.epsilon())

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            sigma = np.dot(flatX.T, flatX) / flatX.shape[0]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)

    def fit_from_directory(self, directory, gt_directory=None, n_classes=None,
                           void_labels=None, cb_weights_method=None):
        """Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            directory: Path to the images
            gt_directory: Path to the masks (Only for segmentation)
            n_classes: Number of classes (Only for segmentation)
            void_labels: Void labels (Only for segmentation)
            cb_weights_method: Class weight balance (Only for segmentation)
        """
        # Get file names
        def get_filenames(directory):
            subdirs = list_subdirs(directory)
            subdirs.append(directory)

            file_names = []
            for subdir in subdirs:
                subpath = os.path.join(directory, subdir)
                for fname in os.listdir(subpath):
                    if has_valid_extension(fname):
                        file_names.append(os.path.join(directory, subdir,
                                                       fname))

            return file_names

        # Precompute the mean and std
        def compute_mean_std(directory, method='mean', mean=None):
            # Get file names
            file_names = get_filenames(directory)

            # Process each file
            sum, n = 0, 0
            for file_name in file_names:
                # Load image and reshape as a vector
                x = io.imread(file_name)
                if self.rescale:
                    x = x*self.rescale
                x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
                n += x.shape[0]

                # Compute mean or std
                if method == 'mean':
                    sum += np.sum(x, axis=0)
                elif method == 'var':
                    x -= mean
                    sum += np.sum(x*x, axis=0)

            return sum/n

        # Compute mean
        if self.featurewise_center:
            self.rgb_mean = compute_mean_std(directory, method='mean',
                                             mean=None)
            # Broadcast the shape
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_index - 1] = len(self.rgb_mean)
            self.mean = np.reshape(self.rgb_mean, broadcast_shape)
            print ('   Mean {}: {}'.format(self.mean.shape, self.rgb_mean,
                                           self.mean))

        # Compute std
        if self.featurewise_std_normalization:
            if not self.featurewise_center:
                self.rgb_mean = compute_mean_std(directory, method='mean',
                                                 mean=None)
            var = compute_mean_std(directory, method='var', mean=self.rgb_mean)
            self.rgb_std = np.sqrt(var)
            # Broadcast the shape
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_index - 1] = len(self.rgb_std)
            self.std = np.reshape(self.rgb_std, broadcast_shape)
            print ('   Std {}: {}'.format(self.std.shape, self.rgb_std))

        # Compute ZCA
        if self.zca_whitening:
            raise ValueError('ZCA Not implemented')

        # Compute class balance segmentation
        if cb_weights_method:
            # Get file names
            file_names = get_filenames(gt_directory)

            # Count the number of samples of each class
            count_per_label = np.zeros(n_classes + len(void_labels))
            total_count_per_label = np.zeros(n_classes + len(void_labels))

            # Process each file
            for file_name in file_names:
                # Load image and mask
                mask = io.imread(file_name)
                mask = mask.astype('int32')

                # Count elements
                unique_labels, counts_label = np.unique(mask, return_counts=True)
                count_per_label[unique_labels] += counts_label
                total_count_per_label[unique_labels] += np.sum(counts_label[:n_classes])

            # Remove void class
            count_per_label = count_per_label[:n_classes]
            total_count_per_label = total_count_per_label[:n_classes]

            # Compute the priors
            priors = count_per_label/total_count_per_label

            # Compute the weights
            self.weights_median_freq_cost = np.median(priors) / priors
            self.weights_rare_freq_cost = 1 / (n_classes * priors)

            # print ('Count per label: ' + str(count_per_label))
            # print ('Total count per label: ' + str(total_count_per_label))
            # print ('Prior: ' + str(priors))
            # print ('Weights median_freq_cost: ' + str(self.weights_median_freq_cost))
            # print ('Weights rare_freq_cost: ' + str(self.weights_rare_freq_cost))

            if cb_weights_method == 'median_freq_cost':
                self.cb_weights = self.weights_median_freq_cost
                print ('Weights median_freq_cost: ' + str(self.weights_median_freq_cost))
            elif cb_weights_method == 'rare_freq_cost':
                self.cb_weights = self.weights_rare_freq_cost
                print ('Weights rare_freq_cost: ' + str(self.weights_rare_freq_cost))
            else:
                raise ValueError('Unknown class balancing method: ' + cb_weights_method)


class DirectoryIterator(Iterator):

    def __init__(self, directory, image_data_generator,
                 resize=None, target_size=None, color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None, gt_directory=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        # Check dim order
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.dim_ordering = dim_ordering

        self.directory = directory
        self.gt_directory = gt_directory
        self.image_data_generator = image_data_generator
        self.resize = resize
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        # Check target size
        if target_size is None and batch_size > 1:
            raise ValueError('Target_size None works only with batch_size=1')
        self.target_size = (None, None) if target_size is None else tuple(target_size)

        # Check color mode
        if color_mode not in {'rgb', 'grayscale', 'bgr'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        if self.color_mode == 'rgb' or self.color_mode == 'bgr':
            self.grayscale = False
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
                self.gt_image_shape = self.target_size + (1,)
            else:
                self.image_shape = (3,) + self.target_size
                self.gt_image_shape = (1,) + self.target_size
        else:
            self.grayscale = True
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
                self.gt_image_shape = self.image_shape
            else:
                self.image_shape = (1,) + self.target_size
                self.gt_image_shape = self.image_shape

        # Check class mode
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'segmentation', 'detection', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "segmentation", "detection" or None.')
        self.class_mode = class_mode
        self.has_gt_image = True if self.class_mode == 'segmentation' else False

        # Check class names
        if not classes:
            if self.class_mode == 'segmentation' or self.class_mode == 'detection':
                raise ValueError('You should input the class names')
            else:
                classes = list_subdirs(directory)
        else:
            classes = classes.values()
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        self.nb_sample = 0
        self.filenames = []
        self.classes = []

        # Get filenames
        if self.class_mode == 'detection':
            for fname in os.listdir(directory):
                if has_valid_extension(fname):
                    self.filenames.append(fname)
                    # Look for the GT filename
                    gt_fname = os.path.join(directory,fname.replace('jpg','txt'))
                    if not os.path.isfile(gt_fname):
                        raise ValueError('GT file not found: ' + gt_fname)
            self.filenames = np.sort(self.filenames)
        elif not self.class_mode == 'segmentation':
            for subdir in classes:
                subpath = os.path.join(directory, subdir)
                for fname in os.listdir(subpath):
                    if has_valid_extension(fname):
                        self.classes.append(self.class_indices[subdir])
                        self.filenames.append(os.path.join(subdir, fname))
            self.classes = np.array(self.classes)
        else:
            for fname in os.listdir(directory):
                if has_valid_extension(fname):
                    self.filenames.append(fname)
                    # Look for the GT filename
                    gt_fname = os.path.join(gt_directory,
                                            os.path.split(fname)[1])
                    if not os.path.isfile(gt_fname):
                        raise ValueError('GT file not found: ' + gt_fname)
            self.filenames = np.sort(self.filenames)

        self.nb_sample = len(self.filenames)
        print('   Found %d images belonging to %d classes' % (self.nb_sample,
                                                            self.nb_class))

        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size,
                                                shuffle, seed)

    def next(self):
        # Lock the generation of index only. The rest is not under thread
        # lock so it can be done in parallel
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        # Create the batch_x and batch_y
        if current_batch_size > 1:
            batch_x = np.zeros((current_batch_size,) + self.image_shape)
            if self.has_gt_image:
                batch_y = np.zeros((current_batch_size,) + self.gt_image_shape)
            if self.class_mode == 'detection':
                batch_y = []

        # Build batch of image data
        for i, j in enumerate(index_array):
            # Load image
            fname = self.filenames[j]
            # print(fname)
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=self.grayscale,
                           resize=self.resize, order=1)
            x = img_to_array(img, dim_ordering=self.dim_ordering)

            # Load GT image if segmentation
            if self.has_gt_image:
                # Load GT image
                gt_img = load_img(os.path.join(self.gt_directory, fname),
                                  grayscale=True,
                                  resize=self.resize, order=0)
                y = img_to_array(gt_img, dim_ordering=self.dim_ordering)
            else:
                y = None

            # Load GT image if detection
            if self.class_mode == 'detection':
                label_path = os.path.join(self.directory, fname).replace('jpg','txt')
                gt = np.loadtxt(label_path)
                if len(gt.shape) == 1:
                    gt = gt[np.newaxis,]
                y = gt.copy()
                y = y[((y[:,1] > 0.) & (y[:,1] < 1.))]
                y = y[((y[:,2] > 0.) & (y[:,2] < 1.))]
                y = y[((y[:,3] > 0.) & (y[:,3] < 1.))]
                y = y[((y[:,4] > 0.) & (y[:,4] < 1.))]
                if (y.shape != gt.shape):
                    warnings.warn('DirectoryIterator: found an invalid annotation '
                                  'on GT file '+label_path)
                # shuffle gt boxes order
                np.random.shuffle(y)


            # Standarize image
            x = self.image_data_generator.standardize(x, y)

            # Data augmentation
            x, y = self.image_data_generator.random_transform(x, y)

            # Add images to batches
            if current_batch_size > 1:
                batch_x[i] = x
                if self.has_gt_image:
                    batch_y[i] = y
                elif self.class_mode == 'detection':
                    batch_y.append(y)
            else:
                batch_x = np.expand_dims(x, axis=0)
                if self.has_gt_image:
                    batch_y = np.expand_dims(y, axis=0)
                elif self.class_mode == 'detection':
                    batch_y = [y]

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):

                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)

                if self.class_mode == 'segmentation':
                    nclasses = self.classes  # TODO: Change
                    color_map = sns.hls_palette(nclasses+1)
                    void_label = nclasses
                    save_img2(batch_x[i], batch_y[i],
                              os.path.join(self.save_to_dir, fname), color_map,
                              void_label)

                else:
                    img = array_to_img(batch_x[i], self.dim_ordering,
                                       scale=True)
                    img.save(os.path.join(self.save_to_dir, fname))

        # Build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        elif self.class_mode == 'detection':
            # YOLOLoss expects a particular batch_y format and shape
            batch_y = yolo_build_gt_batch(batch_y, self.image_shape) 
            # TODO other detection networks may expect a different batch_y format and shape
        elif self.class_mode == None:
            return batch_x

        return batch_x, batch_y


class DirectoryIterator2(object):

    def __init__(self, directory, image_data_generator,
                 resize=None, target_size=None, color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None, gt_directory=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 directory2=None, gt_directory2=None, batch_size2=None):

        self.DI1 = DirectoryIterator(
            directory, image_data_generator, resize=resize,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            gt_directory=gt_directory,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_format=save_format)

        self.DI2 = DirectoryIterator(
            directory2, image_data_generator, resize=resize,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=dim_ordering,
            batch_size=batch_size2, shuffle=shuffle, seed=seed,
            gt_directory=gt_directory2,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_format=save_format)

    def next(self):
        batch_x1, batch_y1 = self.DI1.next()
        batch_x2, batch_y2 = self.DI2.next()
        batch_x = np.concatenate((batch_x1, batch_x2))
        batch_y = np.concatenate((batch_y1, batch_y2))
        return batch_x, batch_y

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
