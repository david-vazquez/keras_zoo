# Import libraries
import numpy as np
import os
import shutil
import skimage.io as io

# Imports
from skimage import img_as_float
import numpy as np
import scipy.misc


# Main function
def compute_class_balance(masks_path, n_classes=4, method='median_freq_cost', void_labels=[5]):

    # Get file names
    file_names = os.listdir(masks_path)

    # Count the number of samples of each class
    count_per_label = np.zeros(n_classes + len(void_labels))
    total_count_per_label = np.zeros(n_classes + len(void_labels))

    # Process each file
    for name in file_names:
        # Load image and mask
        file_name = os.path.join(masks_path, name[:-4] + '.tif')
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
    weights_median_freq_cost = np.median(priors) / priors
    weights_rare_freq_cost = 1 / (n_classes * priors)

    print ('Count per label: ' + str(count_per_label))
    print ('Total count per label: ' + str(total_count_per_label))
    print ('Prior: ' + str(priors))
    print ('Weights median_freq_cost: ' + str(weights_median_freq_cost))
    print ('Weights rare_freq_cost: ' + str(weights_rare_freq_cost))

    if method == 'median_freq_cost':
        return weights_median_freq_cost
    elif method == 'rare_freq_cost':
        return weights_rare_freq_cost
    else:
        print ('ERROR: Unknown class balancing method: ' + method)
        exit()


# Entry point of the script
if __name__ == '__main__':
    weights = compute_class_balance(masks_path='/data/lisa/exp/vazquezd/datasets/polyps_split2/CVC-912/train/masks/',
                                    n_classes=4,
                                    method='median_freq_cost',
                                    void_labels=[5]
                                    )
    print ('Weights   : ' + str(weights))
