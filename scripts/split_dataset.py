# Import libraries
import numpy as np
import os
import shutil


# Get the metadata info from the dataset
def read_csv(file_name, select='frames'):
    # Read the csv data
    from numpy import genfromtxt
    csv_data = genfromtxt(file_name, delimiter=';')
    # print str(my_data)

    if select == 'frames':
        return csv_data, np.unique(csv_data[:, 0]).size
    elif select == 'sequences':
        return csv_data, np.unique(csv_data[:, 3]).size
    elif select == 'patience':
        return csv_data, np.unique(csv_data[:, 1]).size


# Get filenames of the selected ids
def get_names(data, ids, select='frames'):

    # Select elements where the column 'c' is in ids
    def select_elements(data, c, ids):
        select = data[np.logical_or.reduce([data[:, c] == x
                                           for x in ids])].astype(int)
        # print "Select: " + str(select)
        return select

    # Get file names from the selected files
    def select_filenames(select):
        filenames = []
        for i in range(select.shape[0]):
            filenames.append(select[i, 0])
        # print "Filenames: " + str(filenames)
        return filenames

    # Get file names in this frame ids
    if select == 'frames':
        return select_filenames(select_elements(data, 0, ids))
    elif select == 'sequences':
        return select_filenames(select_elements(data, 3, ids))
    elif select == 'patience':
        return select_filenames(select_elements(data, 1, ids))


# Create the output paths
def create_paths(path, set):
    # Create a folder
    def makedir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    # Create path names
    path_set = os.path.join(path, set)
    path_set_images = os.path.join(path_set, "images")
    path_set_gt = os.path.join(path_set, "masks")

    # Create folders
    makedir(path)
    makedir(path_set)
    makedir(path_set_images)
    makedir(path_set_gt)

    # Return folder names
    return path_set, path_set_images, path_set_gt


# Copy images and masks in the folders
def copy_files(filenames, in_image_path, in_mask_path,
               out_image_path, out_mask_path, prefix):
    for i in filenames:
        # Images files
        file_in = os.path.join(in_image_path, str(i) + '.bmp')
        file_out = os.path.join(out_image_path, prefix + '_' + str(i).zfill(3) + '.bmp')
        shutil.copy(file_in, file_out)
        # Mask files
        file_in = os.path.join(in_mask_path, str(i) + '.tif')
        file_out = os.path.join(out_mask_path, prefix + '_' + str(i).zfill(3) + '.tif')
        shutil.copy(file_in, file_out)


# Find the best split of the data
def find_best_split(n, data, split_type, split_prob, max_iters=50000):

    # Decide how many samples to add to each set
    size_set = (n*np.asarray(split_prob)).round()
    size_set[0] = n - (size_set[1] + size_set[2])
    print (' > Num. sets: ' + str(n))
    print (' > Num. sets (train, valid, test): ' + str(size_set))

    # Find iteratively the best data split (Random search)
    best_loss = 9999
    best_split = ()
    best_perc_images = ()
    best_n_images = ()
    for i in range(max_iters):
        # Randomize data
        ids = np.arange(1, n+1)
        np.random.shuffle(ids)

        # Get ids of each set
        ids_train = ids[0:size_set[0]]
        ids_valid = ids[size_set[0]:size_set[0]+size_set[1]]
        ids_test = ids[size_set[0]+size_set[1]:size_set[0]+size_set[1]+size_set[2]]
        # print (' > Sets train: \n' + str(ids_train))
        # print (' > Sets valid: \n' + str(ids_valid))
        # print (' > Sets test:  \n' + str(ids_test))


        # Get filenames
        train_filenames = get_names(data, ids_train, select=split_type)
        valid_filenames = get_names(data, ids_valid, select=split_type)
        test_filenames = get_names(data, ids_test, select=split_type)
        # print (' > Images train: \n' + str(len(train_filenames)))
        # print (' > Images valid: \n' + str(len(valid_filenames)))
        # print (' > Images test:  \n' + str(len(test_filenames)))

        # Count number of images in each set
        n_images = np.asarray((len(train_filenames), len(valid_filenames),
                               len(test_filenames)))
        n_images_total = n_images.sum()
        perc_images = n_images/float(n_images_total)
        loss = (abs(perc_images - split_prob)/split_prob).mean()

        if loss < best_loss:
            best_loss = loss
            best_split = (train_filenames, valid_filenames, test_filenames)
            best_perc_images = perc_images
            best_n_images = n_images

    # Show best split info
    print (' > Total images: ' + str(best_n_images.sum()))
    print (' > Num. images (train, valid, test): ' + str(best_n_images))
    print (' > Per. images (train, valid, test): ' + str(best_perc_images))
    # print (' > Loss: ' + str(best_loss))

    return best_split


# Main function of the script
def split_dataset(in_path='/Tmp/vazquezd/datasets/polyps/CVC-300/',
                  out_path='/Tmp/vazquezd/datasets/polyps_split1/',
                  split_prob=(0.6, 0.2, 0.2),  # (training, validation, test)
                  split_type='frames',  # [frames | sequences | patience]
                  prefix='CVC-300'):

    # Data paths
    image_path = os.path.join(in_path, "bbdd")
    mask_path = os.path.join(in_path, "labelled")
    csv_path = os.path.join(in_path, "data.csv")

    # Read CSV files
    print ('> Reading CSV files...')
    data, n = read_csv(csv_path, select=split_type)

    # Find the best split of the data
    print ('> Creating random split...')
    train_filenames, valid_filenames, test_filenames = find_best_split(n, data, split_type=split_type,
                                                                       split_prob=split_prob)
    # print (' > Images train: \n' + str(train_filenames))
    # print (' > Images valid: \n' + str(valid_filenames))
    # print (' > Images test:  \n' + str(test_filenames))

    # Create the folders
    print ('> Creating output folders...')
    path_train, path_train_images, path_train_gt = create_paths(out_path,
                                                                "train")
    path_valid, path_valid_images, path_valid_gt = create_paths(out_path,
                                                                "valid")
    path_test, path_test_images, path_test_gt = create_paths(out_path, "test")

    # Copy images in the folders
    print ('> Copying the files...')
    copy_files(train_filenames, image_path, mask_path,
               path_train_images, path_train_gt, prefix)
    copy_files(valid_filenames, image_path, mask_path,
               path_valid_images, path_valid_gt, prefix)
    copy_files(test_filenames, image_path, mask_path,
               path_test_images, path_test_gt, prefix)

    print ('> Done!')


# Entry point of the script
if __name__ == '__main__':

    # Pareameters
    in_datasets_path = '/Tmp/vazquezd/datasets/polyps/'
    # out_datasets_path = '/data/lisa/exp/vazquezd/datasets/polyps_split2/'
    out_datasets_path = '/Tmp/vazquezd/datasets/polyps_split/'
    split_prob = (0.6, 0.2, 0.2)  # (training, validation, test)
    split_type = 'patience'  # [frames | sequences | patience]

    # Split the datasets
    print('\n\n ---> Spliting CVC-300 <---')
    split_dataset(in_path=in_datasets_path+'CVC-300/',
                  out_path=out_datasets_path+'CVC-300/',
                  split_prob=split_prob, split_type=split_type,
                  prefix='CVC-300')

    print('\n\n ---> Spliting CVC-612 <---')
    split_dataset(in_path=in_datasets_path+'CVC-612/',
                  out_path=out_datasets_path+'CVC-612/',
                  split_prob=split_prob, split_type=split_type,
                  prefix='CVC-612')

    print('\n\n ---> Spliting CVC-300 (Combined) <---')
    split_dataset(in_path=in_datasets_path+'CVC-300/',
                  out_path=out_datasets_path+'CVC-912/',
                  split_prob=split_prob, split_type=split_type,
                  prefix='CVC-300')

    print('\n\n ---> Spliting CVC-612 (Combined) <---')
    split_dataset(in_path=in_datasets_path+'CVC-612/',
                  out_path=out_datasets_path+'CVC-912/',
                  split_prob=split_prob, split_type=split_type,
                  prefix='CVC-612')
