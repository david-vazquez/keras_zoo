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


# Converts a label mask to RGB to be shown
def my_label2rgb(labels, colors, bglabel=None, bg_color=(0., 0., 0.)):
    output = np.zeros(labels.shape + (3,), dtype=np.float64)
    for i in range(len(colors)):
        if i != bglabel:
            output[(labels == i).nonzero()] = colors[i]
    if bglabel is not None:
        output[(labels == bglabel).nonzero()] = bg_color
    return output


# Converts a label mask to RGB to be shown and overlaps over an image
def my_label2rgboverlay(labels, colors, image, bglabel=None,
                        bg_color=(0., 0., 0.), alpha=0.2):
    image_float = gray2rgb(img_as_float(rgb2gray(image)))
    label_image = my_label2rgb(labels, colors, bglabel=bglabel,
                               bg_color=bg_color)
    output = image_float * alpha + label_image * (1 - alpha)
    return output


# Save image
def save_img(img, mask, result, fname, color_map, void_label):
    img = img / 255.

    label_mask = my_label2rgboverlay(mask,
                                     colors=color_map,
                                     image=img,
                                     bglabel=void_label,
                                     alpha=0.2)

    label_result = my_label2rgboverlay(result,
                                       colors=color_map,
                                       image=img,
                                       bglabel=void_label,
                                       alpha=0.2)

    combined_image = np.concatenate((img, label_mask, label_result), axis=1)
    scipy.misc.toimage(combined_image).save(fname)


# Main function
def main(dataset_path, results_path, n_classes=4):
    # Get paths
    images_path = dataset_path + 'images/'
    masks_path = dataset_path + 'masks/'
    results_cvc300 = results_path + 'CVC-300/labelled/'
    results_cvc612 = results_path + 'CVC-612/labelled/'

    # Get testing files
    file_names = os.listdir(images_path)

    # Create the confusion matrix
    cm = np.zeros((n_classes, n_classes))

    # Process each file
    for name in file_names:
        print (name)
        # Get dataset and file name
        dataset = name[0:7]
        file_raw = name[:-4]
        file_int = int(name[8:-4])

        # Load image, mask and result
        image = io.imread(images_path + name)
        mask = io.imread(masks_path + file_raw + '.tif')
        if dataset == 'CVC-300':
            result = io.imread(results_cvc300 + str(file_int) + '.tif')
        else:
            result = io.imread(results_cvc612 + str(file_int) + '.tif')

        # Make y_true and y_pred flatten
        y_true = mask.flatten()
        y_pred = result.flatten()

        # Fill the confusion matrix
        for i in range(n_classes):
            for j in range(n_classes):
                cm[i, j] += ((y_pred == i) * (y_true == j)).sum()

        # print ('Image shape: ' + str(image.shape))
        # print ('Mask shape: ' + str(mask.shape))
        # print ('Result shape: ' + str(result.shape))

        # Show result
        color_map = sns.hls_palette(5)
        fname = './res.png'
        save_img(image, mask, result, name, color_map, 4)

    # Compute metrics
    TP_perclass = cm.diagonal().astype('float32')
    jaccard_perclass = TP_perclass/(cm.sum(1) + cm.sum(0) - TP_perclass)
    jaccard = np.nanmean(jaccard_perclass)
    accuracy = TP_perclass.sum()/cm.sum()
    print ('Accuracy: ' + str(accuracy))
    print ('Jaccard: ' + str(jaccard))
    print ('Jaccard per class: ' + str(jaccard_perclass))


# Entry point of the script
if __name__ == '__main__':
    main(dataset_path='/data/lisa/exp/vazquezd/datasets/polyps_split2/CVC-912/valid/',
         results_path='/data/lisa/exp/vazquezd/datasets/polypsBaseline2/')
