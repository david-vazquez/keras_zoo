# Import libraries
import numpy as np
import os
import shutil
import skimage.io as io
import seaborn as sns

# Imports
from skimage.color import label2rgb, rgb2gray, gray2rgb
from skimage import img_as_float
import numpy as np
import scipy.misc
from numpy import ma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Main function
def compute(images_path, masks_path, n_classes, method='mean', pre_computed_mean=None,
            pre_computed_std=None):

    # Get testing files
    file_names = os.listdir(images_path)

    total_sum = 0
    total_n = 0
    # min_mask = 999
    # max_mask = 0
    # Process each file
    for name in file_names:
        # Load image and mask
        x = io.imread(os.path.join(images_path, name))
        y = io.imread(os.path.join(masks_path, name[:-4] + '.tif'))

        # Reshape as vectors
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
        y = y.reshape((y.shape[0]*y.shape[1], 1))
        # min_mask = min (min_mask, np.min(y))
        # max_mask = max (max_mask, np.max(y))

        # Substract mean
        if method == 'var':
            x = x - pre_computed_mean

        # Compute the void mask
        mask = np.ones_like(y).astype('int32')
        mask[y == n_classes] = 0.
        mask = np.repeat(mask, 3, axis=1)

        # Mask image
        x_masked = ma.masked_array(x, mask=~mask.astype(bool))

        # Compute mean masked
        if method == 'mean':
            sum_masked = np.sum(x_masked, axis=0)
            n = np.sum(mask, axis=0)

        # Compute std masked
        if method == 'var':
            sum_masked = np.sum(x_masked*x_masked, axis=0)
            n = np.sum(mask, axis=0)

        # Plot
        if method == 'plot':
            # Substract mean
            if pre_computed_mean is not None:
                x = (x - pre_computed_mean)

            # Divide by std
            if pre_computed_std is not None:
                x = x/pre_computed_std

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(x[:, 0], x[:, 1], x[:, 2], c='b', marker='o')
            # plt.show()

            # Plot only one image
            plt.plot(x[:, 0], x[:, 1], 'bo')
            plt.show()
            break

        # Accumulate values
        total_sum += sum_masked
        total_n += n

        # print ('Sum       : ' + str(sum_masked))
        # print ('N         : ' + str(n))
        # print ('Total Sum : ' + str(total_sum))
        # print ('Total N   : ' + str(total_n))
    # print ('Min mask   : ' + str(min_mask))
    # print ('Max mask   : ' + str(max_mask))

    if method != 'plot':
        result = total_sum/total_n
        return result


# Compute mean and std
def compute_mean_std(images_path, masks_path, n_classes):
    mean = compute(images_path, masks_path, n_classes, method='mean',
                   pre_computed_mean=None)
    var = compute(images_path, masks_path, n_classes, method='var',
                  pre_computed_mean=mean)
    std = np.sqrt(var)
    return mean, std


# Main function
def main(images_path, masks_path, n_classes):
    # Compute mean and std
    mean, std = compute_mean_std(images_path, masks_path, n_classes)
    print ('Mean   : ' + str(mean))
    print ('Std   : ' + str(std))

    # Plot before normalization
    # compute(dataset_path, n_classes=4, method='plot')

    # Plot after mean substract
    # compute(dataset_path, n_classes=4, method='plot', pre_computed_mean=mean)

    # Plot after mean substract and divide by std
    # compute(dataset_path, n_classes=4, method='plot', pre_computed_mean=mean,
    #         pre_computed_std=std)


# Entry point of the script
if __name__ == '__main__':
    main('/data/lisa/exp/vazquezd/datasets/polyps_split5/CVC-912/train/images/',
         '/data/lisa/exp/vazquezd/datasets/polyps_split5/CVC-912/train/masks2/',
         2)
