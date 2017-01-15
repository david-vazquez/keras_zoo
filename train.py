#!/usr/bin/env python
# Import python libraries
import (argparse, os)
import sys
import shutil
import time
import imp
import math
from distutils.dir_util import copy_tree
from getpass import getuser
import matplotlib
matplotlib.use('Agg')  # Faster plot
# Import keras libraries
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
from keras.utils.visualize_util import plot
from keras import backend as K

# Import models
from models.fcn8 import build_fcn8
from models.lenet import build_lenet
from models.alexNet import build_alexNet
from models.vgg import build_vgg
from models.resnet import build_resnet50
from models.inceptionV3 import build_inceptionV3

# Import metrics and callbacks
from metrics.metrics import cce_flatt, IoU
from callbacks.callbacks import (History_plot, Jacc_new, Save_results)

# Import tools
from tools.data_loader import ImageDataGenerator
from tools.logger import Logger


# Load datasets
def load_datasets(cf):
    mean = cf.dataset.rgb_mean
    std = cf.dataset.rgb_std
    cf.dataset.cb_weights = None

    # Load training set
    print ('\n > Reading training set...')
    # Create the data generator with its data augmentation
    dg_tr = ImageDataGenerator(imageNet=cf.norm_imageNet_preprocess,
                               rgb_mean=mean,
                               rgb_std=std,
                               rescale=cf.norm_rescale,
                               featurewise_center=cf.norm_featurewise_center,
                               featurewise_std_normalization=cf.norm_featurewise_std_normalization,
                               samplewise_center=cf.norm_samplewise_center,
                               samplewise_std_normalization=cf.norm_samplewise_std_normalization,
                               gcn=cf.norm_gcn,
                               zca_whitening=cf.norm_zca_whitening,
                               crop_size=cf.crop_size_train,
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
                               spline_warp=cf.da_spline_warp,
                               warp_sigma=cf.da_warp_sigma,
                               warp_grid_size=cf.da_warp_grid_size
                               )

    if cf.norm_fit_dataset:
        print ('   Computing normalization constants from training set...')
        # if cf.cb_weights_method is None:
        #     dg_tr.fit_from_directory(cf.dataset.path_train_img)
        # else:
        dg_tr.fit_from_directory(cf.dataset.path_train_img,
                                 cf.dataset.path_train_mask,
                                 len(cf.dataset.classes),
                                 cf.dataset.void_class,
                                 cf.cb_weights_method)

        mean = dg_tr.rgb_mean
        std = dg_tr.rgb_std
        cf.dataset.cb_weights = dg_tr.cb_weights

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
                                          save_to_dir=cf.savepath if cf.da_save_to_dir else None,
                                          save_prefix='data_augmentation',
                                          save_format='png')

    # Load validation set
    print ('\n > Reading validation set...')
    dg_va = ImageDataGenerator(imageNet=cf.norm_imageNet_preprocess,
                               rgb_mean=mean,
                               rgb_std=std,
                               rescale=cf.norm_rescale,
                               featurewise_center=cf.norm_featurewise_center,
                               featurewise_std_normalization=cf.norm_featurewise_std_normalization,
                               samplewise_center=cf.norm_samplewise_center,
                               samplewise_std_normalization=cf.norm_samplewise_std_normalization,
                               gcn=cf.norm_gcn,
                               zca_whitening=cf.norm_zca_whitening)
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
    dg_ts = ImageDataGenerator(imageNet=cf.norm_imageNet_preprocess,
                               rgb_mean=mean,
                               rgb_std=std,
                               rescale=cf.norm_rescale,
                               featurewise_center=cf.norm_featurewise_center,
                               featurewise_std_normalization=cf.norm_featurewise_std_normalization,
                               samplewise_center=cf.norm_samplewise_center,
                               samplewise_std_normalization=cf.norm_samplewise_std_normalization,
                               gcn=cf.norm_gcn,
                               zca_whitening=cf.norm_zca_whitening)
    test_gen = dg_ts.flow_from_directory(directory=cf.dataset.path_test_img,
                                         gt_directory=cf.dataset.path_test_mask,
                                         resize=cf.resize_test,
                                         target_size=cf.target_size_test,
                                         color_mode=cf.dataset.color_mode,
                                         classes=cf.dataset.classes,
                                         class_mode=cf.dataset.class_mode,
                                         batch_size=cf.batch_size_test,
                                         shuffle=cf.shuffle_test,
                                         seed=cf.seed_test)

    # Return the data generators
    return train_gen, valid_gen, test_gen


# Create the optimizer
def create_optimizer(cf):
    # Create the optimizer
    if cf.optimizer == 'rmsprop':
        opt = RMSprop(lr=cf.learning_rate, rho=0.9, epsilon=1e-8, clipnorm=10)
        print ('   Optimizer: rmsprop. Lr: {}. Rho: 0.9, epsilon=1e-8, '
               'clipnorm=10'.format(cf.learning_rate))
    else:
        raise ValueError('Unknown optimizer')

    # Return the optimizer
    return opt


# Build the model
def build_model(cf, optimizer):
    # Define the input size, loss and metrics
    if cf.dataset.class_mode == 'categorical':
        if K.image_dim_ordering() == 'th':
            in_shape = (cf.dataset.n_channels,
                        cf.target_size_train[0],
                        cf.target_size_train[1])
        else:
            in_shape = (cf.target_size_train[0],
                        cf.target_size_train[1],
                        cf.dataset.n_channels)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    elif cf.dataset.class_mode == 'segmentation':
        if K.image_dim_ordering() == 'th':
            in_shape = (cf.dataset.n_channels, None, None)
        else:
            in_shape = (None, None, cf.dataset.n_channels)
        loss = cce_flatt(cf.dataset.void_class, cf.dataset.cb_weights)
        metrics = [IoU(cf.dataset.n_classes, cf.dataset.void_class)]
    else:
        raise ValueError('Unknown problem type')

    # Create the model
    if cf.model_name == 'fcn8':
        model = build_fcn8(in_shape, cf.dataset.n_classes, cf.weight_decay)
    elif cf.model_name == 'lenet':
        model = build_lenet(in_shape, cf.dataset.n_classes, cf.weight_decay)
    elif cf.model_name == 'alexNet':
        model = build_alexNet(in_shape, cf.dataset.n_classes, cf.weight_decay)
    elif cf.model_name == 'vgg16':
        model = build_vgg(in_shape, cf.dataset.n_classes, 16, cf.weight_decay,
                          load_pretrained=cf.load_imageNet,
                          freeze_layers_from=cf.freeze_layers_from)
    elif cf.model_name == 'vgg19':
        model = build_vgg(in_shape, cf.dataset.n_classes, 19, cf.weight_decay,
                          load_pretrained=cf.load_imageNet,
                          freeze_layers_from=cf.freeze_layers_from)
    elif cf.model_name == 'resnet50':
        model = build_resnet50(in_shape, cf.dataset.n_classes, cf.weight_decay,
                               load_pretrained=cf.load_imageNet,
                               freeze_layers_from=cf.freeze_layers_from)
    elif cf.model_name == 'InceptionV3':
        model = build_inceptionV3(in_shape, cf.dataset.n_classes,
                                  cf.weight_decay,
                                  load_pretrained=cf.load_imageNet,
                                  freeze_layers_from=cf.freeze_layers_from)
    else:
        raise ValueError('Unknown model')

    # Load pretrained weights
    if cf.load_pretrained:
        # Get weights file name
        path, _ = os.path.split(cf.weights_file)
        if path == '':
            weights_file = os.path.join(cf.savepath, cf.weights_file)
        else:
            weights_file = cf.weights_file

        # Load weights
        print('   loading model weights from: ' + weights_file + '...')
        model.load_weights(weights_file, by_name=True)

    # Compile model
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    # Show model structure
    if cf.show_model:
        model.summary()
        plot(model, to_file=os.path.join(cf.savepath, 'model.png'))

    # Output the model
    print ('   Model: ' + cf.model_name)
    return model


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


# Predict the model
def predict_model(cf, model, test_gen):
    if cf.test_model:
        print('\n > Testing the model...')
        # Load best trained model
        model.load_weights(os.path.join(cf.savepath, "weights.hdf5"))

        start_time = time.time()
        model.predict_generator(test_gen, cf.n_images_test,
                                max_q_size=10, nb_worker=1, pickle_safe=False)
        total_time = time.time() - start_time
        fps = float(cf.n_images_test) / total_time
        s_p_f = total_time / float(cf.n_images_test)
        print ('   Testing time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time, fps, s_p_f))


# Test the model
def test_model(cf, model, test_gen):
    if cf.test_model:
        print('\n > Testing the model...')
        # Load best trained model
        model.load_weights(os.path.join(cf.savepath, "weights.hdf5"))

        start_time = time.time()
        test_metrics = model.evaluate_generator(test_gen,
                                                cf.n_images_test,
                                                max_q_size=10,
                                                nb_worker=1,
                                                pickle_safe=False)
        total_time = time.time() - start_time
        fps = float(cf.n_images_test) / total_time
        s_p_f = total_time / float(cf.n_images_test)
        print ('   Testing time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time, fps, s_p_f))
        print ('   Testing metrics: {}'.format(model.metrics_names))
        print ('   Testing metrics: {}'.format(test_metrics))


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

    # Create the data generators
    train_gen, valid_gen, test_gen = load_datasets(cf)

    # Create the optimizer
    print ('\n > Creating optimizer...')
    optimizer = create_optimizer(cf)

    # Build model
    print ('\n > Building model...')
    model = build_model(cf, optimizer)

    # Create the callbacks
    print ('\n > Creating callbacks...')
    cb = create_callbacks(cf, valid_gen)

    # Train the model
    hist = train_model(cf, model, train_gen, valid_gen, cb)

    # Compute test metrics
    test_metrics = test_model(cf, model, test_gen)

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
    if cf.debug and cf.debug_images_train > 0:
        cf.dataset.n_images_train = cf.debug_images_train
    if cf.debug and cf.debug_images_valid > 0:
        cf.dataset.n_images_valid = cf.debug_images_valid
    if cf.debug and cf.debug_images_test > 0:
        cf.dataset.n_images_test = cf.debug_images_test

    # Define target sizes
    if cf.crop_size_train is not None:cf.target_size_train = cf.crop_size_train
    elif cf.resize_train is not None:cf.target_size_train = cf.resize_train
    else:cf.target_size_train = cf.dataset.img_shape

    if cf.crop_size_valid is not None:cf.target_size_valid = cf.crop_size_valid
    elif cf.resize_valid is not None:cf.target_size_valid = cf.resize_valid;
    else:cf.target_size_valid = cf.dataset.img_shape

    if cf.crop_size_test is not None:cf.target_size_test = cf.crop_size_test
    elif cf.resize_test is not None:cf.target_size_test = cf.resize_test
    else:cf.target_size_test = cf.dataset.img_shape

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

    assert arguments.config_path is not None, 'Please provide a configuration'\
                                              'path using -c config/pathname'\
                                              ' in the command line'
    assert arguments.exp_name is not None, 'Please provide a name for the '\
                                           'experiment using -e name in the '\
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
