# Import python libraries
import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Import keras libraries
from keras.callbacks import (EarlyStopping,
                             ModelCheckpoint)
from keras.optimizers import RMSprop
from keras.utils.visualize_util import plot

# Import project libraries
from fcn8 import build_fcn8
from loader_sem_seg import ImageDataGenerator
from metrics import cce_flatt
from callbacks.callbacks import ValJaccard


# Save the printf to a log file
class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", 0)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# Train the network
def train(dataset, model_name, learning_rate, weight_decay,
          num_epochs, max_patience, batch_size, optimizer,
          savepath, train_path, valid_path, test_path,
          crop_size=(224, 224), in_shape=(3, None, None), n_classes=5,
          load_weights=False, void_class=[4]):

    # Remove void classes from number of classes
    n_classes = n_classes - len(void_class)

    # TODO: Get the number of images directly from data loader
    n_images_train = 547  # 547
    n_images_val = 183  # 183
    n_images_test = 182  # 182

    # Build model
    print ' > Building model...'
    if model_name == 'fcn8':
        model = build_fcn8(in_shape, regularize_weights=weight_decay,
                           nclasses=n_classes, load_weights=load_weights)
    else:
        raise ValueError('Unknown model')

    # TODO: What is this??
    model.output

    # Compile model
    print ' > Compiling model...'
    optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-8, clipnorm=10)
    model.compile(loss=cce_flatt(void_class), optimizer=optimizer)
    # TODO: Add metrics: Jaccard and DICE

    # Show model structure
    model.summary()
    plot(model, to_file=savepath+'model.png')

    # Create the data generators
    print ('\n > Reading training set...')
    dg_tr = ImageDataGenerator(crop_size=crop_size,  # Crop the image to a fixed size
                               featurewise_center=False,  # Set input mean to 0 over the dataset
                               samplewise_center=False,  # Set each sample mean to 0
                               featurewise_std_normalization=False,  # Divide inputs by std of the dataset
                               samplewise_std_normalization=False,  # Divide each input by its std
                               zca_whitening=False,  # Apply ZCA whitening
                               rotation_range=180,  # Randomly rotate images in the range (degrees, 0 to 180)
                               width_shift_range=0.0,  # Randomly shift images horizontally (fraction of total width)
                               height_shift_range=0.0,  # Randomly shift images vertically (fraction of total height)
                               shear_range=0.,  # Shear Intensity (Shear angle in counter-clockwise direction as radians)
                               zoom_range=0.1,  # Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range]
                               channel_shift_range=0.,  # Range for random channel shifts.
                               fill_mode='constant',  # One of {"constant", "nearest", "reflect" or "wrap"}. Points outside the boundaries of the input are filled according to the given mode.
                               cval=0.,  # Value used for points outside the boundaries when fill_mode = "constant".
                               cvalMask=n_classes,  # Void class value
                               horizontal_flip=True,  # Randomly flip images horizontally
                               vertical_flip=True, # Randomly flip images vertically
                               rescale=None,  # Rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation).
                               spline_warp=True,
                               warp_sigma=0.1,
                               warp_grid_size=3
                               )
    train_gen = dg_tr.flow_from_directory(train_path + 'images',
                                          batch_size=batch_size,
                                          gt_directory=train_path + 'masks',
                                          target_size=crop_size,
                                          class_mode='seg_map',
                                          classes=n_classes,
                                          #save_to_dir=savepath,
                                          save_prefix='data_augmentation',
                                          save_format='png')

    print ('\n > Reading validation set...')
    dg_va = ImageDataGenerator()
    valid_gen = dg_va.flow_from_directory(valid_path + 'images',
                                          batch_size=1,
                                          gt_directory=valid_path + 'masks',
                                          target_size=crop_size,
                                          class_mode='seg_map',
                                          classes=n_classes)

    print ('\n > Reading testing set...')
    dg_ts = ImageDataGenerator()
    test_gen = dg_ts.flow_from_directory(test_path + 'images',
                                         batch_size=1,
                                         gt_directory=test_path + 'masks',
                                         target_size=crop_size,
                                         class_mode='seg_map',
                                         classes=n_classes)

    # Define the jaccard validation callback
    val_jaccard = ValJaccard(nclasses=n_classes, valid_gen=valid_gen,
                             metrics=['val_loss', 'val_jaccard', 'val_acc',
                                      'val_jaccard_perclass'],
                             epoch_length=n_images_val,
                             void_label=void_class[0],
                             out_images_folder=savepath)

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_jaccard', mode='max',
                                   patience=max_patience, verbose=0)

    # Define model saving callback
    checkpointer = ModelCheckpoint(filepath=savepath+"weights.hdf5", verbose=1,
                                   monitor='val_jaccard', mode='max',
                                   save_best_only=True)

    # Train the model
    print('\n > Training the model...')
    hist = model.fit_generator(train_gen, samples_per_epoch=n_images_train,
                               nb_epoch=num_epochs,
                               # validation_data=valid_gen,
                               # nb_val_samples=n_images_val,
                               callbacks=[val_jaccard, early_stopping,
                                          checkpointer])

    # Show the trained model history
    print('\n > Show the trained model history...')
    print(hist.history.keys())
    plt.plot(hist.history['loss'],
             label='train loss ({:.3f})'.format(np.min(hist.history['loss'])))
    plt.plot(hist.history['val_loss'],
             label='valid loss ({:.3f})'.format(np.min(hist.history['val_loss'])))
    plt.plot(hist.history['val_jaccard'],
             label='validation jaccard ({:.3f})'.format(np.max(hist.history['val_jaccard'])))
    plt.plot(hist.history['val_acc'],
             label='validation accuracy ({:.3f})'.format(np.max(hist.history['val_acc'])))

    # Add title
    plt.title('Model training history')

    # Add axis labels
    plt.ylabel('Metric')
    plt.xlabel('Epoch')

    # Add legend
    plt.legend(loc='upper left')

    # Save fig
    plt.savefig(savepath+'plot.png')

    # plt.legend(['train loss', 'valid loss', 'valid jaccard'], loc='upper left')

    # Show plot
    plt.show()


# Main function
def main():
    # Get parameters from file parser
    parser = argparse.ArgumentParser(description='Unet model training')
    parser.add_argument('-dataset', default='polyps', help='Dataset')
    parser.add_argument('-model_name', default='fcn8', help='Model')
    parser.add_argument('-learning_rate', default=0.0001, help='Learning Rate')
    parser.add_argument('-weight_decay', default=0.0,
                        help='regularization constant')
    parser.add_argument('--num_epochs', '-ne', type=int, default=1000,
                        help='Optional. Int to indicate the max'
                        'number of epochs.')
    parser.add_argument('-max_patience', type=int, default=100,
                        help='Max patience (early stopping)')
    parser.add_argument('-batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--optimizer', '-opt', default='rmsprop',
                        help='Optimizer')
    args = parser.parse_args()

    # Michail paths
    # savepath = '/home/michal/tmp',
    # train_path = '/home/michal/polyps/CVC-612/'
    # valid_path = '/home/michal/polyps/CVC-300/')
    # test_path = '/home/michal/polyps/CVC-300/')

    # David paths
    savepath = '/Tmp/vazquezd/results/deepPolyp/fcn8/tmpDA2/'
    dataset_path = '/Tmp/vazquezd/datasets/polyps_split2/CVC-912/'
    train_path = dataset_path + 'train/'
    valid_path = dataset_path + 'valid/'
    test_path = dataset_path + 'test/'

    # Create output folders
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Enable log file
    sys.stdout = Logger(savepath + "logfile.log")

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
          train_path=train_path, valid_path=valid_path, test_path=test_path,
          crop_size=(224, 224), in_shape=(3, None, None), n_classes=5,
          load_weights=False,
          void_class=[4])

# Entry point of the script
if __name__ == "__main__":
    main()
