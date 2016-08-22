# Import libraries
import argparse
from keras.callbacks import (EarlyStopping,
                             ModelCheckpoint)
from keras.optimizers import RMSprop
from fcn8 import build_fcn8
from loader_sem_seg import ImageDataGenerator
from metrics import cce_flatt, jaccard
from callbacks.callbacks import ValJaccard


# Train the network
def train(dataset, model_name, learning_rate, weight_decay,
          num_epochs, max_patience, batch_size, optimizer='rmsprop',
          savepath='/home/michal/tmp/',
          train_path='/home/michal/polyps/CVC-612/',
          val_path='/home/michal/polyps/CVC-300/',
          crop_size=(224, 224), in_shape=(3, None, None), n_classes=5,
          load_weights=False, void_class=[4]):

    # Remove void classes from number of classes
    n_classes = n_classes - len(void_class)

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

    # Data augmentation methods
    dg_tr = ImageDataGenerator(crop_size=crop_size)
    dg_ts = ImageDataGenerator()

    # Create the data generators
    train_gen = dg_tr.flow_from_directory(train_path + 'images',
                                          batch_size=10,
                                          gt_directory=train_path + 'masks',
                                          target_size=crop_size,
                                          class_mode='seg_map')

    valid_gen = dg_ts.flow_from_directory(val_path + 'images',
                                          batch_size=1,
                                          gt_directory=val_path + 'masks',
                                          target_size=crop_size,
                                          class_mode='seg_map')

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=50,
                                   verbose=0, mode='auto')

    # Define model saving callback
    checkpointer = ModelCheckpoint(filepath="./weights.hdf5", verbose=1,
                                   save_best_only=True)

    # Define the jaccard callback
    val_jaccard = ValJaccard(nclasses=n_classes, valid_gen=valid_gen,
                             epoch_length=183, void_label=void_class[0])  # TODO: Define size

    print(' > Training the model...')
    hist = model.fit_generator(train_gen, samples_per_epoch=547, nb_epoch=2,  # TODO: Define size
                               validation_data=valid_gen, nb_val_samples=183,  # TODO: Define size
                               callbacks=[early_stopping, checkpointer,
                                          val_jaccard]
                               )


# Main function
def main():
    # Get parameters from file parser
    parser = argparse.ArgumentParser(description='Unet model training')
    parser.add_argument('-dataset', default='camvid', help='Dataset')
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
    # val_path = '/home/michal/polyps/CVC-300/')

    # David paths
    savepath = '/Tmp/vazquezd/results/deepPolyp/'
    train_path = '/Tmp/vazquezd/datasets/polyps_split2/CVC-912/train/'
    val_path = '/Tmp/vazquezd/datasets/polyps_split2/CVC-912/valid/'

    # Train the network
    train(args.dataset, args.model_name, float(args.learning_rate),
          float(args.weight_decay), int(args.num_epochs),
          int(args.max_patience), int(args.batch_size),  args.optimizer,
          savepath=savepath, train_path=train_path, val_path=val_path,
          crop_size=(224, 224))


# Entry point of the script
if __name__ == "__main__":
    main()
