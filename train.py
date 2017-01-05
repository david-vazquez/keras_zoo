#!/usr/bin/env python
# Import python libraries
import argparse
import os
import sys
import shutil
import time
import imp
import pickle
import math
from distutils.dir_util import copy_tree
from getpass import getuser
import matplotlib
matplotlib.use('Agg')  # Faster plot

# Import keras libraries
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
from keras.utils.visualize_util import plot

# Import project libraries
from models.fcn8 import build_fcn8
from models.alexNet import build_alexNet
from metrics.metrics import cce_flatt, IoU
from callbacks.callbacks import Evaluate_model, compute_metrics, History_plot, Jacc_new, Save_results
from tools.loader_sem_seg import ImageDataGenerator
from tools.logger import Logger
from tools.plot_history import plot_history
from tools.compute_mean_std import compute_mean_std
from tools.compute_class_balance import compute_class_balance


# Normalization mean and std computed on training set for RGB pixel values
def compute_normalization_constants(cf):
    if cf.input_norm in {'mean', 'std', 'meanAndStd'}:
        if cf.compute_constants:
             m, s = compute_mean_std(cf.dataset.path_train_img,
                                     cf.dataset.path_train_mask,
                                     cf.dataset.n_classes)
             cf.dataset.rgb_mean, cf.dataset.rgb_std = m, s
        if cf.input_norm not in {'mean', 'meanAndStd'}:
            cf.dataset.rgb_mean = None
        if cf.input_norm not in {'std', 'meanAndStd'}:
            cf.dataset.rgb_std = None
        cf.dataset.rgb_rescale = None
    elif cf.input_norm == 'rescale':
        if cf.compute_constants:
            cf.dataset.rgb_rescale = 1/255.
        cf.dataset.rgb_mean, cf.dataset.rgb_std = None, None
    else:
        raise ValueError('Unknown normalization scheme')


# Compute class balance weights
def compute_class_balance_weights(cf):
    if cf.cb_weights_method is not None:
        w = compute_class_balance(cf.dataset.path_train_mask,
                                  n_classes=cf.dataset.n_classes,
                                  method=cf.cb_weights_method,
                                  void_labels=cf.dataset.void_class)
        cf.dataset.cb_weights = w
    else:
        cf.dataset.cb_weights = None


# Create the optimizer
def create_optimizer(cf):
    # Create the optimizer
    if cf.optimizer == 'rmsprop':
        opt = RMSprop(lr=cf.learning_rate, rho=0.9, epsilon=1e-8, clipnorm=10)
        print ('   Optimizer: rmsprop. Lr: {}. Rho: 0.9, epsilon=1e-8,' \
               ' clipnorm=10'.format(cf.learning_rate))
    else:
        raise ValueError('Unknown optimizer')

    # Return the optimizer
    return opt


# Build the model
def build_model(cf, optimizer):
    # Get pretrained weights if needed
    weights_file = os.path.join(cf.savepath, cf.weights_file) if cf.load_pretrained else False

    # Create the model

    if cf.model_name == 'fcn8':
        in_shape = (cf.dataset.n_channels, None, None)
        model = build_fcn8(in_shape,
                           l2_reg=cf.weight_decay,
                           nclasses=cf.dataset.n_classes,
                           weights_file=weights_file)
        # Compile
        model.compile(loss=cce_flatt(cf.dataset.void_class, cf.dataset.cb_weights),
                      metrics=[IoU(cf.dataset.n_classes, cf.dataset.void_class)],
                      optimizer=optimizer)
    elif cf.model_name == 'alexNet':
        in_shape = (cf.dataset.n_channels, cf.target_size_train[0], cf.target_size_train[1])
        model = build_alexNet(in_shape,
                              l2_reg=cf.weight_decay,
                              n_classes=cf.dataset.n_classes,
                              weights_file=weights_file)
        # Compile
        model.compile(loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      optimizer=optimizer)
    else:
        raise ValueError('Unknown model')

    # Show model structure
    if cf.show_model:
        model.summary()
        plot(model, to_file=os.path.join(cf.savepath, 'model.png'))

    # Output the model
    print ('   Model: ' + cf.model_name)
    return model


# Load datasets
def load_datasets(cf):
    # Load training set
    print ('\n > Reading training set...')
    # Create the data generator with its data augmentation
    dg_tr = ImageDataGenerator(crop_size=cf.crop_size_train,
                               featurewise_center=cf.da_featurewise_center,
                               samplewise_center=cf.da_samplewise_center,
                               featurewise_std_normalization=cf.da_featurewise_std_normalization,
                               samplewise_std_normalization=cf.da_samplewise_std_normalization,
                               rgb_mean=cf.dataset.rgb_mean,
                               rgb_std=cf.dataset.rgb_std,
                               gcn=cf.da_gcn,
                               zca_whitening=cf.da_zca_whitening,
                               rotation_range=cf.da_rotation_range,
                               width_shift_range=cf.da_width_shift_range,
                               height_shift_range=cf.da_height_shift_range,
                               shear_range=cf.da_shear_range,
                               zoom_range=cf.da_zoom_range,
                               channel_shift_range=cf.da_channel_shift_range,
                               fill_mode=cf.da_fill_mode,
                               cval=cf.da_cval,
                               void_label=cf.dataset.void_class[0] if cf.dataset.void_class else None,
                               horizontal_flip=cf.da_horizontal_flip,
                               vertical_flip=cf.da_vertical_flip,
                               rescale=cf.dataset.rgb_rescale,
                               spline_warp=cf.da_spline_warp,
                               warp_sigma=cf.da_warp_sigma,
                               warp_grid_size=cf.da_warp_grid_size
                               )
    train_gen = dg_tr.flow_from_directory(directory=cf.dataset.path_train_img,
                                          gt_directory=cf.dataset.path_train_mask,
                                          resize=cf.resize_train,
                                          target_size=cf.target_size_train,
                                          color_mode=cf.dataset.color_mode,
                                          classes=cf.dataset.classes,
                                          class_mode=cf.dataset.class_mode,
                                          batch_size=cf.batch_size_train,
                                          shuffle=cf.shuffle_train,
                                          seed=cf.seed_train,
                                          save_to_dir=savepath if cf.da_save_to_dir else None,
                                          save_prefix='data_augmentation',
                                          save_format='png')

    # Load validation set
    print ('\n > Reading validation set...')
    dg_va = ImageDataGenerator(rgb_mean=cf.rgb_mean, rgb_std=cf.rgb_std,
                               rescale=cf.rgb_rescale)
    valid_gen = dg_va.flow_from_directory(directory=cf.dataset.path_valid_img,
                                          gt_directory=cf.dataset.path_valid_mask,
                                          resize=cf.resize_valid,
                                          target_size=cf.target_size_valid,
                                          color_mode=cf.dataset.color_mode,
                                          classes=cf.dataset.classes,
                                          class_mode=cf.dataset.class_mode,
                                          batch_size=cf.batch_size_valid,
                                          shuffle=cf.shuffle_valid,
                                          seed=cf.seed_valid)

    # Load testing set
    print ('\n > Reading testing set...')
    dg_ts = ImageDataGenerator(rgb_mean=cf.rgb_mean, rgb_std=cf.rgb_std,
                               rescale=cf.rgb_rescale)
    test_gen = dg_ts.flow_from_directory(directory=cf.dataset.path_test_img,
                                         gt_directory=cf.dataset.path_test_mask,
                                         resize=cf.resize_test,
                                         target_size=cf.target_size_test,
                                         color_mode=cf.dataset.color_mode,
                                         classes=cf.dataset.classes, #cf.dataset.n_classes,
                                         class_mode=cf.dataset.class_mode,
                                         batch_size=cf.batch_size_test,
                                         shuffle=cf.shuffle_test,
                                         seed=cf.seed_test)

    # Return the data generators
    return train_gen, valid_gen, test_gen


# Create callbacks
def create_callbacks(cf, valid_gen):

    cb = []

    # Jaccard callback
    if cf.dataset.class_mode == 'segmentation':
        print('   Jaccard metric')
        cb += [Jacc_new(cf.dataset.n_classes)]

    # Save image results
    if cf.save_results_enabled:
        print('   Save image result')
        cb += [Save_results(n_classes=cf.dataset.n_classes,
                            void_label=cf.dataset.void_class,
                            save_path=cf.dataset.savepath,
                            generator=valid_gen,
                            epoch_length=int(math.ceil(cf.save_results_nsamples/float(cf.save_results_batch_size))),
                            color_map=cf.dataset.color_map,
                            tag='valid')]

    # Early stopping
    if cf.earlyStopping_enabled:
        print('   Early stopping')
        cb += [EarlyStopping(monitor=cf.earlyStopping_monitor,
                             mode=cf.earlyStopping_mode,
                             patience=cf.earlyStopping_patience,
                             verbose=cf.earlyStopping_verbose)]

    # Define model saving callbacks
    if cf.checkpoint_enabled:
        print('   Model Checkpoint')
        cb += [ModelCheckpoint(filepath=os.path.join(cf.savepath, "weights.hdf5"),
                               verbose=cf.checkpoint_verbose,
                               monitor=cf.checkpoint_monitor,
                               mode=cf.checkpoint_mode,
                               save_best_only=cf.checkpoint_save_best_only,
                               save_weights_only=cf.checkpoint_save_weights_only)]

    # Plot the loss after every epoch.
    if cf.plotHist_enabled:
        print('   Plot per epoch')
        cb += [History_plot(cf.dataset.n_classes, cf.savepath,
                            cf.train_metrics, cf.valid_metrics,
                            cf.best_metric, cf.best_type, cf.plotHist_verbose)]

    # # Define the jaccard validation callback
    # eval_model = Evaluate_model(n_classes=cf.dataset.n_classes,
    #                             void_label=cf.dataset.void_class[0],
    #                             save_path=cf.savepath,
    #                             valid_gen=valid_gen,
    #                             valid_epoch_length=cf.n_images_valid/cf.batch_size_valid,
    #                             valid_metrics=cf.valid_metrics,
    #                             color_map=cf.dataset.color_map)
    # print('   Jaccard validation callback')
    #
    # early_stop_jac_class = []
    # for i in range(cf.dataset.n_classes):
    #     early_stop_jac_class += [EarlyStopping(monitor=str(i)+'_val_jacc',
    #                                            mode=cf.earlyStopping_mode,
    #                                            patience=cf.earlyStopping_patience,
    #                                            verbose=cf.earlyStopping_verbose)]
    #
    #
    # checkp_jac_class = []
    # for i in range(cf.dataset.n_classes):
    #     checkp_jac_class += [ModelCheckpoint(filepath=os.path.join(cf.savepath, "weights"+str(i)+".hdf5"),
    #                                          verbose=cf.checkpoint_verbose,
    #                                          monitor=str(i)+'_val_jacc',
    #                                          mode=cf.checkpoint_mode,
    #                                          save_best_only=cf.checkpoint_save_best_only,
    #                                          save_weights_only=cf.checkpoint_save_weights_only)]
    #

    # Output the list of callbacks
    return cb


# Train the model
def train_model(cf, model, train_gen, valid_gen, cb):
    if (cf.train_model):
        print('\n > Training the model...')
        hist = model.fit_generator(generator=train_gen,
                                   samples_per_epoch=cf.n_images_train,
                                   nb_epoch=cf.n_epochs,
                                   verbose=1,
                                   callbacks=cb,
                                   validation_data=valid_gen,
                                   nb_val_samples=cf.n_images_valid,
                                   class_weight=None,
                                   max_q_size=10,
                                   nb_worker=1,
                                   pickle_safe=False)
        print('   Training finished.')

        return hist
    else:
        return None


# Test the model
def test_model(cf, model, test_gen):
    if cf.test_model:
        print('\n > Testing the model...')
        # Load best trained model
        model.load_weights(os.path.join(cf.savepath, "weights.hdf5"))
        # Compute metrics
        test_metrics = compute_metrics(model, test_gen, cf.dataset.n_images_test,
                                       cf.dataset.n_classes,
                                       metrics=['test_loss',
                                                'test_jaccard',
                                                'test_acc',
                                                'test_jaccard_perclass'],
                                       color_map=cf.color_map,
                                       tag="test",
                                       void_label=cf.dataset.void_class[0],
                                       out_images_folder=cf.savepath,
                                       epoch=0,
                                       save_all_images=True
                                       )
        # Show results
        for k in sorted(test_metrics.keys()):
            print('   {}: {}'.format(k, test_metrics[k]))

        # return metrics
        return test_metrics
    else:
        return None


# Plot training history
def plot_training_history(cf, hist, test_metrics):
    if (hist is not None):
        # Save the results
        print ("\n > Saving history...")
        with open(os.path.join(cf.savepath, "history.pickle"), 'w') as f:
            pickle.dump([hist.history, test_metrics], f)

        # Show the trained model history
        if cf.plot_hist:
            # Load the results
            print ("\n > Loading history...")
            with open(os.path.join(cf.savepath, "history.pickle")) as f:
                history, test_metrics = pickle.load(f)

            # Plot history
            print('\n > Ploting the trained model history...')
            #plot_history(history, cf.savepath, cf.dataset.n_classes)
            plot_history(history, cf.savepath, cf.dataset.n_classes,
                         train_metrics=cf.train_metrics,
                         valid_metrics=cf.valid_metrics,
                         best_metric=cf.best_metric,
                         best_type=cf.best_type,
                         verbose=True)

# Copy result to shared directory
def copy_to_shared(cf):
    if cf.savepath != cf.final_savepath:
        print('\n > Copying model and other training files to {}'.format(cf.final_savepath))
        start = time.time()
        copy_tree(cf.savepath, cf.final_savepath)
        open(os.path.join(cf.final_savepath, 'lock'), 'w').close()
        print ('   Copy time: ' + str(time.time()-start))


# Train the network
def train(cf):

    # Enable log file
    sys.stdout = Logger(cf.log_file)
    print (' ---> Init experiment: ' + cf.exp_name + ' <---')

    # Compute normalization constants
    print ('\n > Computing normalization constants...')
    compute_normalization_constants(cf)
    print ('   Mean: {}. Std: {}. Rescale: {}.'.format(cf.dataset.rgb_mean,
                                                       cf.dataset.rgb_std,
                                                       cf.dataset.rgb_rescale))

    # Compute class balance weights
    print ('\n > Computing class balance weights...')
    compute_class_balance_weights(cf)
    print ('   Weights: ' + str(cf.dataset.cb_weights))

    # Create the optimizer
    print ('\n > Creating optimizer...')
    optimizer = create_optimizer(cf)

    # Build model
    print ('\n > Building model...')
    model = build_model(cf, optimizer)

    # Create the data generators
    train_gen, valid_gen, test_gen = load_datasets(cf)

    # Create the callbacks
    print ('\n > Creating callbacks...')
    cb = create_callbacks(cf, valid_gen)

    # Train the model
    hist = train_model(cf, model, train_gen, valid_gen, cb)

    # Compute test metrics
    test_metrics = test_model(cf, model, test_gen)

    # Plot training history
    plot_training_history(cf, hist, test_metrics)

    # Copy result to shared directory
    copy_to_shared(cf)

    # Finish
    print (' ---> Finish experiment: ' + cf.exp_name + ' <---')


def load_config_files(config_path, exp_name,
                      dataset_path, shared_dataset_path,
                      experiments_path, shared_experiments_path):

    # Load configuration file
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

    # Copy the dataset from the shared to the local path if not existing
    shared_dataset_path = os.path.join(shared_dataset_path, cf.dataset_name)
    dataset_path = os.path.join(dataset_path, cf.dataset_name)
    if not os.path.exists(dataset_path):
        print('The local path {} does not exist. Copying '
              'dataset...'.format(dataset_path))
        shutil.copytree(shared_dataset_path, dataset_path)
        print('Done.')

    # Load dataset
    dataset_config_path = os.path.join(dataset_path, 'config.py')
    cf.dataset = imp.load_source('config', dataset_config_path)
    cf.dataset.config_path = dataset_config_path

    # Compose dataset paths
    cf.dataset.path = dataset_path
    if cf.dataset.class_mode == 'segmentation':
        cf.dataset.path_train_img = os.path.join(cf.dataset.path, 'train', 'images')
        cf.dataset.path_train_mask = os.path.join(cf.dataset.path, 'train', 'masks')
        cf.dataset.path_valid_img = os.path.join(cf.dataset.path, 'valid', 'images')
        cf.dataset.path_valid_mask = os.path.join(cf.dataset.path, 'valid', 'masks')
        cf.dataset.path_test_img = os.path.join(cf.dataset.path, 'test', 'images')
        cf.dataset.path_test_mask = os.path.join(cf.dataset.path, 'test', 'masks')
    else:
        cf.dataset.path_train_img = os.path.join(cf.dataset.path, 'train')
        cf.dataset.path_train_mask = None
        cf.dataset.path_valid_img = os.path.join(cf.dataset.path, 'valid')
        cf.dataset.path_valid_mask = None
        cf.dataset.path_test_img = os.path.join(cf.dataset.path, 'test')
        cf.dataset.path_test_mask = None

    # If in Debug mode use few images
    if cf.debug and cf.debug_images_train>0:
        cf.dataset.n_images_train = cf.debug_images_train
    if cf.debug and cf.debug_images_valid>0:
        cf.dataset.n_images_valid = cf.debug_images_valid
    if cf.debug and cf.debug_images_test>0:
        cf.dataset.n_images_test = cf.debug_images_test

    # Define target sizes
    if cf.crop_size_train is not None: cf.target_size_train = cf.crop_size_train
    elif cf.resize_train is not None: cf.target_size_train = cf.resize_train
    else: cf.target_size_train = cf.dataset.img_shape

    if cf.crop_size_valid is not None: cf.target_size_valid = cf.crop_size_valid
    elif cf.resize_valid is not None: cf.target_size_valid = cf.resize_valid;
    else: cf.target_size_valid = cf.dataset.img_shape

    if cf.crop_size_test is not None: cf.target_size_test = cf.crop_size_test
    elif cf.resize_test is not None: cf.target_size_test = cf.resize_test
    else: cf.target_size_test = cf.dataset.img_shape

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

    return cf


# Main function
def main():

    # Define the user paths
    usr_path = os.path.join('/home', getuser())
    shared_path = '/datatmp'
    dataset_path = os.path.join(shared_path, 'Datasets')
    shared_dataset_path = os.path.join(shared_path, 'Datasets')
    experiments_path = os.path.join(shared_path, 'Experiments')
    shared_experiments_path = os.path.join(shared_path, 'Experiments')

    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='FCN model training')
    parser.add_argument('-c', '--config_path',
                        type=str,
                        default=None,
                        help='Configuration file')
    parser.add_argument('-e', '--exp_name',
                        type=str,
                        default=None,
                        help='Name of the experiment')
    arguments = parser.parse_args()

    assert arguments.config_path is not None, 'Please provide a configuration' \
                                              'path using -c config/pathname' \
                                              ' in the command line'
    assert arguments.exp_name is not None, 'Please provide a name for the ' \
                                           'experiment using -e name in the ' \
                                           'command line'

    # Load configuration files
    cf = load_config_files(arguments.config_path, arguments.exp_name,
                           dataset_path, shared_dataset_path,
                           experiments_path, shared_experiments_path)

    # Train the network
    train(cf)


# Entry point of the script
if __name__ == "__main__":
    main()
