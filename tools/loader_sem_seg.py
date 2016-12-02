from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from numpy import ma
from six.moves import range
import os
import seaborn as sns
import scipy.misc
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


# Load image
def load_img(path, grayscale=False, target_size=None):
    import skimage.io as io
    from skimage.color import rgb2gray

    # Load image
    img = io.imread(path)

    # # Resize
    # print('target_size: ' + str(target_size))
    # if target_size[0] is not None:
    #     raise ValueError('Not implemented')

    # Convert to grayscale
    if grayscale:
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
                 gcn=False,
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
                 spline_warp=False,
                 warp_sigma=0.1,
                 warp_grid_size=3,
                 dim_ordering='default',
                 rgb_mean=None,
                 rgb_std=None,
                 crop_size=None):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.rescale = rescale

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

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_format=save_format)

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            gt_directory=None,
                            save_to_dir=None, save_prefix='',
                            save_format='jpeg'):
        return DirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            gt_directory=gt_directory,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_format=save_format)

    def standardize(self, x, y=None):
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1

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
            # print ('Mean   : ' + str(mean))
            # print ('Std   : ' + str(std))

            # Normalize
            s = 1
            eps = 1e-8
            x = s * (x - mean_masked) / max(eps, std_masked)

            # Set void pixels to 0
            x = x*mask

            # max_v = np.max(x)
            # min_v = np.min(x)
            # x_after = (x-min_v)/(max_v-min_v)
            # combined_image = np.concatenate((x_before, x_after), axis=1)
            # import scipy.misc
            # scipy.misc.toimage(combined_image).save('./gcn.png')
            # exit()

        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.rgb_mean is not None:
            x -= np.reshape(self.rgb_mean, (3, 1, 1))
        if self.rgb_std is not None:
            x /= np.reshape(self.rgb_std, (3, 1, 1))

        if self.zca_whitening:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
        return x

    def random_transform(self, x, y=None):
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        # use composition of homographies to generate final transform that
        # needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range,
                                                    self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range,
                                   self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range,
                                   self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0],
                                       self.zoom_range[1], 2)
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
            y = apply_transform(y, transform_matrix, img_channel_index,
                                fill_mode=self.fill_mode, cval=self.void_label)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range,
                                     img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                if y is not None:
                    y = flip_axis(y, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                if y is not None:
                    y = flip_axis(y, img_row_index)

        if self.spline_warp:
            warp_field = gen_warp_field(shape=x.shape[-2:],
                                        sigma=self.warp_sigma,
                                        grid_size=self.warp_grid_size)
            x = apply_warp(x, warp_field,
                           interpolator=sitk.sitkLinear,
                           fill_mode=self.fill_mode, fill_constant=self.cval)

            if y is not None:
                y = np.round(apply_warp(y, warp_field,
                                        interpolator=sitk.sitkNearestNeighbor,
                                        fill_mode=self.fill_mode,
                                        fill_constant=self.void_label))

        # Crop
        # TODO: tf compatible???
        crop = list(self.crop_size) if self.crop_size else None
        if crop:
            # print ('X before: ' + str(x.shape))
            # print ('Y before: ' + str(y.shape))
            # print ('Crop_size: ' + str(self.crop_size))
            h, w = x.shape[img_row_index], x.shape[img_col_index]

            # Padd image if it is smaller than the crop size
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
                y = np.lib.pad(y, ((0, 0), (pad_h1, pad_h2), (pad_w1, pad_w2)),
                               'constant', constant_values=self.void_label)
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
                    y = y[..., :, top:top+crop[0], left:left+crop[1]]
            else:
                x = x[..., top:top+crop[0], left:left+crop[1], :]
                if y is not None:
                    y = y[..., top:top+crop[0], left:left+crop[1], :]

            # print ('X after: ' + str(x.shape))
            # print ('Y after: ' + str(y.shape))

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        if y is None:
            return x
        else:
            return x, y

    def fit(self, X, augment=False, rounds=1, seed=None):
        '''Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        # Arguments
            X: Numpy array, the data to fit on.
            augment: whether to fit on randomly augmented samples
            rounds: if `augment`, how many augmentation passes to do over
            the data
            seed: random seed.
        '''
        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= (self.std + 1e-7)

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)


class DirectoryIterator(Iterator):

    def __init__(self, directory, image_data_generator,
                 target_size=None, color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None, gt_directory=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
        self.gt_directory = gt_directory
        self.image_data_generator = image_data_generator
        if target_size is None and batch_size > 1:
            raise ValueError('Target_size None works only with batch_size=1')
        self.target_size = (None, None) if target_size is None else tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'seg_map', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "seg_map" or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'tif'}

        # first, count the number of samples and classes
        self.nb_sample = 0
        self.nb_GT_sample = 0
        self.filenames = []
        self.gt_filenames = []
        if not self.class_mode == 'seg_map':
            if not classes:
                classes = []
                for subdir in sorted(os.listdir(directory)):
                    if os.path.isdir(os.path.join(directory, subdir)):
                        classes.append(subdir)
            self.nb_class = len(classes)
            self.class_indices = dict(zip(classes, range(len(classes))))

            for subdir in classes:
                subpath = os.path.join(directory, subdir)
                for fname in os.listdir(subpath):
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.nb_sample += 1
            print('Found %d images belonging to %d classes.' % (self.nb_sample,
                                                                self.nb_class))
        else:
            for fname in os.listdir(directory):
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.filenames.append(fname)
                    self.nb_sample += 1
            print('Found %d images.' % (self.nb_sample))
            self.filenames = np.sort(self.filenames)
            for fname in os.listdir(gt_directory):
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.gt_filenames.append(fname)
                    self.nb_GT_sample += 1
            print('Found %d GT images.' % (self.nb_sample))
            self.gt_filenames = np.sort(self.gt_filenames)

        if not self.class_mode == 'seg_map':
            self.classes = np.zeros((self.nb_sample,), dtype='int32')
            i = 0
            for subdir in classes:
                subpath = os.path.join(directory, subdir)
                for fname in os.listdir(subpath):
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.classes[i] = self.class_indices[subdir]
                        self.filenames.append(os.path.join(subdir, fname))
                        i += 1

        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size,
                                                shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can
        # be done in parallel
        if self.target_size[0] is None:
            batch_x = 0
        else:
            batch_x = np.zeros((current_batch_size,) + self.image_shape)
        if self.class_mode == 'seg_map':
            if self.target_size[0] is None:
                batch_y = 0
            else:
                batch_y = np.zeros((current_batch_size,
                                    1, self.image_shape[1],
                                    self.image_shape[2]))
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            gtname = self.gt_filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            if not self.class_mode == 'seg_map':
                x = self.image_data_generator.standardize(x)
                x = self.image_data_generator.random_transform(x)
                if current_batch_size > 1:
                    batch_x[i] = x
                else:
                    batch_x = np.expand_dims(x, axis=0)
            else:
                GT = load_img(os.path.join(self.gt_directory, gtname),
                              grayscale=False,
                              target_size=self.target_size)
                y = img_to_array(GT, dim_ordering=self.dim_ordering)
                x = self.image_data_generator.standardize(x, y)
                x, y = self.image_data_generator.random_transform(x, y)
                if current_batch_size > 1:
                    batch_x[i] = x
                    batch_y[i] = y
                else:
                    batch_x = np.expand_dims(x, axis=0)
                    batch_y = np.expand_dims(y, axis=0)
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):

                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)

                if self.class_mode == 'seg_map':
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
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        elif not self.class_mode == 'seg_map':
            return batch_x
        # print(self.directory, np.shape(batch_x), np.shape(batch_y))
        return batch_x, batch_y
