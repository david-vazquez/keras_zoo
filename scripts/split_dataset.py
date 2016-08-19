# Import libraries
import random
import numpy as np
import os


# Parameters
split_prob = (0.6, 0.2, 0.2)  # (training, validation, test)
split_type = 'frames'  # [frames | sequences | patience]

# Data paths
shared_path = '/data/lisa/exp/vazquezd/datasets/polyps/'
path = '/Tmp/vazquezd/datasets/polyps/'
image_path_300 = os.path.join(path, 'CVC-300', "bbdd")
mask_path_300 = os.path.join(path, 'CVC-300', "labelled")
csv_path_300 = os.path.join(shared_path, "CVC-300.csv")
image_path_612 = os.path.join(path, 'CVC-612', "bbdd")
mask_path_612 = os.path.join(path, 'CVC-612', "labelled")
csv_path_612 = os.path.join(shared_path, "CVC-612.csv")


# Get the metadata info from the dataset
def read_csv(file_name):
    from numpy import genfromtxt
    csv_data = genfromtxt(file_name, delimiter=';')
    return csv_data
    print str(my_data)


# Read CSV files
CVC_300_data = read_csv(csv_path_300)
CVC_612_data = read_csv(csv_path_612)

# Split the data randomly
ids_300 = random.shuffle(range(300))
ids_612 = random.shuffle(range(612))

n_samples_split = np.random.multinomial(300, np.asarray(split_prob), size=None)
print ('n_samples_split' + str(n_samples_split))





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
            filenames.append(str(select[i, 0]))
        # print "Filenames: " + str(filenames)
        return filenames

    # Get file names in this frame ids
    if select == 'frames':
        return select_filenames(select_elements(data, 0, ids))
    elif select == 'sequences':
        return select_filenames(select_elements(data, 3, ids))
    elif select == 'patience':
        return select_filenames(select_elements(data, 1, ids))

#
#
# def get_names(select='frames'):
#
#     # Select elements where the column 'c' is in ids
#     def select_elements(data, c, ids):
#         select = data[np.logical_or.reduce([data[:, c] == x
#                                            for x in ids])].astype(int)
#         # print "Select: " + str(select)
#         return select
#
#     # Get file names from the selected files
#     def select_filenames(select):
#         filenames = []
#         for i in range(select.shape[0]):
#             filenames.append(str(select[i, 0]))
#         # print "Filenames: " + str(filenames)
#         return filenames
#
#     # Get file names in this frame ids
#     def by_frame(data, ids):
#         return select_filenames(select_elements(data, 0, ids))
#
#     # Get file names in this sequence ids
#     def by_sequence(data, ids):
#         return select_filenames(select_elements(data, 3, ids))
#
#     # Get file names in this sequence ids
#     def by_patience(data, ids):
#         return select_filenames(select_elements(data, 1, ids))
#
#     def get_file_names_by_frame(data, id_first, id_last):
#         return by_frame(data, range(id_first, id_last))
#
#     def get_file_names_by_sequence(data, id_first, id_last):
#         return by_sequence(data, range(id_first, id_last))
#
#     def get_file_names_by_patience(data, id_first, id_last):
#         return by_patience(data, range(id_first, id_last))
#
#     # Get the metadata info from the dataset
#     def read_csv(file_name):
#         from numpy import genfromtxt
#         csv_data = genfromtxt(file_name, delimiter=';')
#         return csv_data
#         print str(my_data)
#         # [Frame ID, Patiend ID, Polyp ID, Polyp ID2]
#
#     if select == 'frames':
#         # Get file names for this set
#         if self.which_set == "train":
#             self.filenames = get_file_names_by_frame(self.CVC_612_data,
#                                                      1, 401)
#             self.is_300 = False
#         elif self.which_set == "val":
#             self.filenames = get_file_names_by_frame(self.CVC_612_data,
#                                                      401, 501)
#             self.is_300 = False
#         elif self.which_set == "test":
#             self.filenames = get_file_names_by_frame(self.CVC_612_data,
#                                                      501, 613)
#             self.is_300 = False
#         else:
#             print 'EROR: Incorret set: ' + self.filenames
#             exit()
#     elif self.select == 'sequences':
#         # Get file names for this set
#         if self.which_set == "train":
#             self.filenames = get_file_names_by_sequence(self.CVC_612_data,
#                                                         1, 21)
#             self.is_300 = False
#         elif self.which_set == "val":
#             self.filenames = get_file_names_by_sequence(self.CVC_612_data,
#                                                         21, 26)
#             self.is_300 = False
#         elif self.which_set == "test":
#             self.filenames = get_file_names_by_sequence(self.CVC_612_data,
#                                                         26, 32)
#             self.is_300 = False
#         else:
#             print 'EROR: Incorret set: ' + self.filenames
#             exit()
#     elif self.select == 'patience':
#         # Get file names for this set
#         if self.which_set == "train":
#             self.filenames = get_file_names_by_patience(self.CVC_612_data,
#                                                         1, 16)
#             self.is_300 = False
#         elif self.which_set == "val":
#             self.filenames = get_file_names_by_patience(self.CVC_612_data,
#                                                         16, 21)
#             self.is_300 = False
#         elif self.which_set == "test":
#             self.filenames = get_file_names_by_patience(self.CVC_612_data,
#                                                         21, 26)
#             self.is_300 = False
#         else:
#             print 'EROR: Incorret set: ' + self.filenames
#             exit()
#     else:
#         print 'EROR: Incorret select: ' + self.select
#         exit()
#
#     # Load CSV files
