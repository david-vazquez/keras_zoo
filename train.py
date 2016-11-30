# Import python libraries
import argparse
import os
import sys
import seaborn as sns
from getpass import getuser
import shutil
import numpy as np

# Import keras libraries
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
from keras.utils.visualize_util import plot

# Import project libraries
from models.fcn8 import build_fcn8
from metrics.metrics import cce_flatt
from callbacks.callbacks import Evaluate_model, compute_metrics
from tools.loader_sem_seg import ImageDataGenerator
from tools.logger import Logger
from tools.plot_history import plot_history
from models.model import assemble_model
from tools.compute_mean_std import compute_mean_std
from tools.compute_class_balance import compute_class_balance
sys.setrecursionlimit(99999)


# Train the network
def train(dataset, model_name, learning_rate, weight_decay,
          num_epochs, max_patience, batch_size, optimizer,
          savepath, train_path, valid_path, test_path,
          crop_size=(224, 224), in_shape=(3, None, None), n_classes=5, gtSet=1,
          weights_file=False, void_class=[4], show_model=False,
          plot_hist=True):

    # Remove void classes from number of classes
    n_classes = n_classes - len(void_class)

    # TODO: Get the number of images directly from data loader
    n_images_train = 547  # 547
    n_images_val = 183  # 183
    n_images_test = 182  # 182

    # Normalization mean and std computed on training set for RGB pixel values
    print '\n > Computing mean and std for normalization...'
    if False:
        rgb_mean, rgb_std = compute_mean_std(train_path, n_classes)
        # rgb_mean = np.asarray([136.80214937481, 89.02750787575, 60.9570439560])
        # rgb_std = np.asarray([61.55742495180, 49.316179114493, 38.362239487371])
    else:
        rgb_mean = None
        rgb_std = None
    print ('Mean: ' + str(rgb_mean))
    print ('Std: ' + str(rgb_std))

    # Compute class balance weights
    if True:
        class_balance_weights = compute_class_balance(masks_path=train_path + 'masks' + str(gtSet),
                                                      n_classes=n_classes,
                                                      method='median_freq_cost',
                                                      void_labels=void_class
                                                      )
    else:
        class_balance_weights = None
    print ('Class balance weights: ' + str(class_balance_weights))

    # Build model
    print '\n > Building model (' + model_name + ')...'
    if model_name == 'fcn8':
        model = build_fcn8(in_shape, l2_reg=weight_decay, nclasses=n_classes,
                           weights_file=weights_file, deconv='deconv')
        model.output
    elif model_name == 'resunet':
        model_kwargs = {
            'input_shape': in_shape,
            'num_classes': n_classes,
            'input_num_filters': 32,
            'main_block_depth': [3, 8, 10, 3],
            'num_main_blocks': 3,
            'num_init_blocks': 1,
            'W_regularizer': weight_decay,
            'dropout': 0.2,
            'short_skip': True,
            'long_skip': True,
            'use_skip_blocks': False,
            'relative_num_across_filters': 1,
            'long_skip_merge_mode': 'sum'}
        model = assemble_model(**model_kwargs)
    else:
        raise ValueError('Unknown model')

    # Create the optimizer
    print '\n > Creating optimizer ({}) with lr ({})...'.format(optimizer,
                                                                learning_rate)
    if optimizer == 'rmsprop':
        opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-8, clipnorm=10)
    else:
        raise ValueError('Unknown optimizer')

    # Compile model
    print '\n > Compiling model...'
    model.compile(loss=cce_flatt(void_class, class_balance_weights),
                  optimizer=opt)

    # Show model structure
    if show_model:
        model.summary()
        plot(model, to_file=savepath+'model.png')

    # Create the data generators
    print ('\n > Reading training set...')
    dg_tr = ImageDataGenerator(crop_size=crop_size,  # Crop the image to a fixed size
                               featurewise_center=False,  # Substract mean - dataset
                               samplewise_center=False,  # Substract mean - sample
                               featurewise_std_normalization=False,  # Divide std - dataset
                               samplewise_std_normalization=False,  # Divide std - sample
                               rgb_mean=rgb_mean,
                               rgb_std=rgb_std,
                               gcn=False,  # Global contrast normalization
                               zca_whitening=False,  # Apply ZCA whitening
                               rotation_range=0,  # Rnd rotation degrees 0-180
                               width_shift_range=0.0,  # Rnd horizontal shift
                               height_shift_range=0.0,  # Rnd vertical shift
                               shear_range=0.,  # 0.5,  # Shear in radians
                               zoom_range=0.,  # Zoom
                               channel_shift_range=0.,  # Channel shifts
                               fill_mode='constant',  # Fill mode
                               cval=0.,  # Void image value
                               void_label=void_class[0],  # Void class value
                               horizontal_flip=False,  # Rnd horizontal flip
                               vertical_flip=False,  # Rnd vertical flip
                               rescale=None,  # Rescaling factor
                               spline_warp=False,  # Enable elastic deformation
                               warp_sigma=10,  # Elastic deformation sigma
                               warp_grid_size=3  # Elastic deformation gridSize
                               )
    train_gen = dg_tr.flow_from_directory(train_path + 'images',
                                          batch_size=batch_size,
                                          gt_directory=train_path + 'masks' + str(gtSet),
                                          target_size=crop_size,
                                          class_mode='seg_map',
                                          classes=n_classes,
                                          # save_to_dir=savepath,  # Save DA
                                          save_prefix='data_augmentation',
                                          save_format='png')

    print ('\n > Reading validation set...')
    dg_va = ImageDataGenerator(rgb_mean=rgb_mean, rgb_std=rgb_std)
    valid_gen = dg_va.flow_from_directory(valid_path + 'images',
                                          batch_size=1,
                                          gt_directory=valid_path + 'masks' + str(gtSet),
                                          target_size=None,
                                          class_mode='seg_map',
                                          classes=n_classes)

    print ('\n > Reading testing set...')
    dg_ts = ImageDataGenerator(rgb_mean=rgb_mean, rgb_std=rgb_std)
    test_gen = dg_ts.flow_from_directory(test_path + 'images',
                                         batch_size=1,
                                         gt_directory=test_path + 'masks' + str(gtSet),
                                         target_size=None,
                                         class_mode='seg_map',
                                         classes=n_classes)

    # Define the jaccard validation callback
    evaluate_model = Evaluate_model(n_classes=n_classes,
                                    void_label=void_class[0],
                                    save_path=savepath,
                                    valid_gen=valid_gen,
                                    valid_epoch_length=n_images_val,
                                    valid_metrics=['val_loss',
                                                   'val_jaccard',
                                                   'val_acc',
                                                   'val_jaccard_perclass'])

    # Define early stopping callback
    # TODO: Make a for
    early_stopping_jaccard = EarlyStopping(monitor='val_jaccard', mode='max',
                                           patience=max_patience, verbose=0)
    early_stopping_jaccard_0 = EarlyStopping(monitor='0_val_jacc_percl',
                                             mode='max', patience=max_patience,
                                             verbose=0)
    early_stopping_jaccard_1 = EarlyStopping(monitor='1_val_jacc_percl',
                                             mode='max', patience=max_patience,
                                             verbose=0)
    early_stopping_jaccard_2 = EarlyStopping(monitor='2_val_jacc_percl',
                                             mode='max', patience=max_patience,
                                             verbose=0)
    early_stopping_jaccard_3 = EarlyStopping(monitor='3_val_jacc_percl',
                                             mode='max', patience=max_patience,
                                             verbose=0)
    early_stopping_jaccard_4 = EarlyStopping(monitor='4_val_jacc_percl',
                                             mode='max', patience=max_patience,
                                             verbose=0)

    # Define model saving callback
    # TODO: Make a for
    checkpointer_jaccard = ModelCheckpoint(filepath=savepath+"weights.hdf5",
                                           verbose=0, monitor='val_jaccard',
                                           mode='max', save_best_only=True,
                                           save_weights_only=True)
    checkpointer_jaccard_0 = ModelCheckpoint(filepath=savepath+"weights0.hdf5",
                                             verbose=0,
                                             monitor='0_val_jacc_percl',
                                             mode='max', save_best_only=True,
                                             save_weights_only=True)
    checkpointer_jaccard_1 = ModelCheckpoint(filepath=savepath+"weights1.hdf5",
                                             verbose=0,
                                             monitor='1_val_jacc_percl',
                                             mode='max', save_best_only=True,
                                             save_weights_only=True)
    checkpointer_jaccard_2 = ModelCheckpoint(filepath=savepath+"weights2.hdf5",
                                             verbose=0,
                                             monitor='2_val_jacc_percl',
                                             mode='max', save_best_only=True,
                                             save_weights_only=True)
    checkpointer_jaccard_3 = ModelCheckpoint(filepath=savepath+"weights3.hdf5",
                                             verbose=0,
                                             monitor='3_val_jacc_percl',
                                             mode='max', save_best_only=True,
                                             save_weights_only=True)
    checkpointer_jaccard_4 = ModelCheckpoint(filepath=savepath+"weights4.hdf5",
                                             verbose=0,
                                             monitor='4_val_jacc_percl',
                                             mode='max', save_best_only=True,
                                             save_weights_only=True)

    # Train the model
    print('\n > Training the model...')
    # TODO: Make a for
    if n_classes == 5:
        cb = [evaluate_model, early_stopping_jaccard, checkpointer_jaccard, checkpointer_jaccard_0,
              checkpointer_jaccard_1, checkpointer_jaccard_2, checkpointer_jaccard_3, checkpointer_jaccard_4]
    elif n_classes == 4:
        cb = [evaluate_model, early_stopping_jaccard, checkpointer_jaccard, checkpointer_jaccard_0,
              checkpointer_jaccard_1, checkpointer_jaccard_2, checkpointer_jaccard_3]
    elif n_classes == 3:
        cb = [evaluate_model, early_stopping_jaccard, checkpointer_jaccard, checkpointer_jaccard_0,
              checkpointer_jaccard_1, checkpointer_jaccard_2]
    elif n_classes == 2:
        cb = [evaluate_model, early_stopping_jaccard, checkpointer_jaccard, checkpointer_jaccard_0,
              checkpointer_jaccard_1]
    else:
        raise ValueError('Incorrect number of classes')

    hist = model.fit_generator(train_gen, samples_per_epoch=n_images_train,
                               nb_epoch=num_epochs,
                               callbacks=cb)

    # Compute test metrics
    print('\n > Testing the model...')
    model.load_weights(savepath + "weights.hdf5")
    color_map = sns.hls_palette(n_classes+1)
    test_metrics = compute_metrics(model, test_gen, n_images_test, n_classes,
                                   metrics=['test_loss',
                                            'test_jaccard',
                                            'test_acc',
                                            'test_jaccard_perclass'],
                                   color_map=color_map, tag="test",
                                   void_label=void_class[0],
                                   out_images_folder=savepath,
                                   epoch=0,
                                   save_all_images=True,
                                   useCRF=False)

    for k in sorted(test_metrics.keys()):
        print('{}: {}'.format(k, test_metrics[k]))

    # Show the trained model history
    if plot_hist:
        print('\n > Show the trained model history...')
        plot_history(hist, savepath)


# Main function
def main():
    # Get parameters from file parser
    parser = argparse.ArgumentParser(description='DeepPolyp model training')
    parser.add_argument('-dataset', default='polyps', help='Dataset')
    parser.add_argument('-model_name', default='fcn8', help='Model')
    parser.add_argument('-model_file', default='weights.hdf5',
                        help='Model file')
    parser.add_argument('-load_pretrained', default=False,
                        help='Load pretrained model from model file')
    parser.add_argument('-learning_rate', default=0.001, help='Learning Rate')
    parser.add_argument('-weight_decay', default=0.,
                        help='regularization constant')
    parser.add_argument('--num_epochs', '-ne', type=int, default=1000,
                        help='Optional. Int to indicate the max'
                        'number of epochs.')
    parser.add_argument('-max_patience', type=int, default=50,
                        help='Max patience (early stopping)')
    parser.add_argument('-batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--optimizer', '-opt', default='rmsprop',
                        help='Optimizer')
    args = parser.parse_args()

    # Experiment name
    experiment_name = "tmpWeightBalance"  #### Pay attention ####

    # Define paths according to user
    usr = getuser()
    if usr == "michal":
        # Michal paths
        savepath = '/home/michal/' + experiment_name + '/'
        dataset_path = '/home/michal/polyps/polyps_split2/CVC-912/'
        train_path = dataset_path + 'train/'
        valid_path = dataset_path + 'valid/'
        test_path = dataset_path + 'test/'
    elif usr == 'vazquezd' or usr == 'romerosa':
        shared_dataset_path = '/data/lisa/exp/vazquezd/datasets/polyps_split5/CVC-912/'
        dataset_path = '/Tmp/'+usr+'/datasets/polyps_split5/CVC-912/'
        # Copy the data to the local path if not existing
        if not os.path.exists(dataset_path):
            print('The local path {} does not exist. Copying '
                  'dataset...'.format(dataset_path))
            shutil.copytree(shared_dataset_path, dataset_path)
            print('Done.')

        savepath = '/Tmp/'+usr+'/results/deepPolyp/fcn8/'+experiment_name+'/'
        train_path = dataset_path + 'train/'
        valid_path = dataset_path + 'valid/'
        test_path = dataset_path + 'test/'

    elif usr == 'dvazquez':
        shared_dataset_path = '/home/'+usr+'/Datasets/Polyps/'
        dataset_path = '/home/'+usr+'/Datasets/Polyps/'
        # Copy the data to the local path if not existing
        if not os.path.exists(dataset_path):
            print('The local path {} does not exist. Copying '
                  'dataset...'.format(dataset_path))
            shutil.copytree(shared_dataset_path, dataset_path)
            print('Done.')

        savepath = '/home/'+usr+'/Experiments/deepPolyp/'+experiment_name+'/'
        train_path = dataset_path + 'train/'
        valid_path = dataset_path + 'valid/'
        test_path = dataset_path + 'test/'

    else:
        raise ValueError('User unknown, please add your own paths!')

    # Create output folders
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Enable log file
    sys.stdout = Logger(savepath + "logfile.log")
    print ' ---> Experiment: ' + experiment_name + ' <---'

    # Train the network
    train(dataset=args.dataset,
          model_name=args.model_name,
          learning_rate=float(args.learning_rate),
          weight_decay=float(args.weight_decay),
          num_epochs=int(args.num_epochs),
          max_patience=int(args.max_patience),
          batch_size=int(args.batch_size),
          optimizer=args.optimizer,
          savepath=savepath,
          show_model=False,
          train_path=train_path, valid_path=valid_path, test_path=test_path,
          crop_size=(224, 224), in_shape=(3, None, None),
          n_classes=3,  #### Pay attention ####
          gtSet=5,  #### Pay attention ####
          weights_file=savepath+args.model_file if bool(args.load_pretrained) else False,
          void_class=[2])  #### Pay attention ####
    print ' ---> Experiment: ' + experiment_name + ' <---'

# Entry point of the script
if __name__ == "__main__":
    main()
