import imp
import time
import os
from distutils.dir_util import copy_tree
import shutil


class Configuration():
    def __init__(self, config_path, exp_name,
                       dataset_path, shared_dataset_path,
                       experiments_path, shared_experiments_path):

        self.config_path = config_path
        self.exp_name = exp_name
        self.dataset_path = dataset_path
        self.shared_dataset_path = shared_dataset_path
        self.experiments_path = experiments_path
        self.shared_experiments_path = shared_experiments_path

    def load(self):
        config_path = self.config_path
        exp_name = self.exp_name
        dataset_path = self.dataset_path
        shared_dataset_path = self.shared_dataset_path
        experiments_path = self.experiments_path
        shared_experiments_path = self.shared_experiments_path

        # Load configuration file
        print config_path
        cf = imp.load_source('config', config_path)

        # Save extra parameter
        cf.config_path = config_path
        cf.exp_name = exp_name

        # Create output folders
        cf.savepath = os.path.join(experiments_path, cf.dataset_name, cf.exp_name)
        cf.final_savepath = os.path.join(shared_experiments_path, cf.dataset_name,
                                         cf.exp_name)
        cf.log_file = os.path.join(cf.savepath, "logfile.log")
        if not os.path.exists(cf.savepath):
            os.makedirs(cf.savepath)

        # Copy config file
        shutil.copyfile(config_path, os.path.join(cf.savepath, "config.py"))

        # Load dataset configuration
        cf.dataset = self.load_config_dataset(cf.dataset_name, dataset_path,
                                              shared_dataset_path,
                                              cf.problem_type,
                                              'config_dataset')
        if cf.dataset_name2:
            cf.dataset2 = self.load_config_dataset(cf.dataset_name2,
                                                   dataset_path,
                                                   shared_dataset_path,
                                                   cf.problem_type,
                                                   'config_dataset2')

        # If in Debug mode use few images
        if cf.debug and cf.debug_images_train > 0:
            cf.dataset.n_images_train = cf.debug_images_train
        if cf.debug and cf.debug_images_valid > 0:
            cf.dataset.n_images_valid = cf.debug_images_valid
        if cf.debug and cf.debug_images_test > 0:
            cf.dataset.n_images_test = cf.debug_images_test
        if cf.debug and cf.debug_n_epochs > 0:
            cf.n_epochs = cf.debug_n_epochs

        # Define target sizes
        if cf.crop_size_train is not None: cf.target_size_train = cf.crop_size_train
        elif cf.resize_train is not None: cf.target_size_train = cf.resize_train
        else: cf.target_size_train = cf.dataset.img_shape

        if cf.crop_size_valid is not None: cf.target_size_valid = cf.crop_size_valid
        elif cf.resize_valid is not None: cf.target_size_valid = cf.resize_valid
        else: cf.target_size_valid = cf.dataset.img_shape

        if cf.crop_size_test is not None: cf.target_size_test = cf.crop_size_test
        elif cf.resize_test is not None: cf.target_size_test = cf.resize_test
        else: cf.target_size_test = cf.dataset.img_shape

        # Get weights file name
        path, _ = os.path.split(cf.weights_file)
        if path == '':
            cf.weights_file = os.path.join(cf.savepath, cf.weights_file)

        # Plot metrics
        if cf.dataset.class_mode == 'segmentation':
            cf.train_metrics = ['loss', 'acc', 'jaccard']
            cf.valid_metrics = ['val_loss', 'val_acc', 'val_jaccard']
            cf.best_metric = 'val_jaccard'
            cf.best_type = 'max'
        else:
            cf.train_metrics = ['loss', 'acc']
            cf.valid_metrics = ['val_loss', 'val_acc']
            cf.best_metric = 'val_acc'
            cf.best_type = 'max'

        self.configuration = cf
        return cf

    # Load the configuration file of the dataset
    def load_config_dataset(self, dataset_name, dataset_path, shared_dataset_path, problem_type, name='config'):
        # Copy the dataset from the shared to the local path if not existing
        shared_dataset_path = os.path.join(shared_dataset_path, problem_type, dataset_name)
        dataset_path = os.path.join(dataset_path, problem_type, dataset_name)
        if not os.path.exists(dataset_path):
            print('The local path {} does not exist. Copying '
                  'dataset...'.format(dataset_path))
            shutil.copytree(shared_dataset_path, dataset_path)
            print('Done.')

        # Load dataset config file
        dataset_config_path = os.path.join(dataset_path, 'config.py')
        print 'dataset_config_path', dataset_config_path
        dataset_conf = imp.load_source(name, dataset_config_path)
        dataset_conf.config_path = dataset_config_path

        # Compose dataset paths
        dataset_conf.path = dataset_path
        if dataset_conf.class_mode == 'segmentation':
            dataset_conf.path_train_img = os.path.join(dataset_conf.path, 'train', 'images')
            dataset_conf.path_train_mask = os.path.join(dataset_conf.path, 'train', 'masks')
            dataset_conf.path_valid_img = os.path.join(dataset_conf.path, 'valid', 'images')
            dataset_conf.path_valid_mask = os.path.join(dataset_conf.path, 'valid', 'masks')
            dataset_conf.path_test_img = os.path.join(dataset_conf.path, 'test', 'images')
            dataset_conf.path_test_mask = os.path.join(dataset_conf.path, 'test', 'masks')
        else:
            dataset_conf.path_train_img = os.path.join(dataset_conf.path, 'train')
            dataset_conf.path_train_mask = None
            dataset_conf.path_valid_img = os.path.join(dataset_conf.path, 'valid')
            dataset_conf.path_valid_mask = None
            dataset_conf.path_test_img = os.path.join(dataset_conf.path, 'test')
            dataset_conf.path_test_mask = None

        return dataset_conf

    # Copy result to shared directory
    def copy_to_shared(self):
        if self.configuration.savepath != self.configuration.final_savepath:
            print('\n > Copying model and other training files to {}'.format(self.configuration.final_savepath))
            start = time.time()
            copy_tree(self.configuration.savepath, self.configuration.final_savepath)
            open(os.path.join(self.configuration.final_savepath, 'lock'), 'w').close()
            print ('   Copy time: ' + str(time.time()-start))
