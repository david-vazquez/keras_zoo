import argparse

from keras.callbacks import (EarlyStopping,
                             ModelCheckpoint)

from fcn8 import build_fcn8
from metrics import categorical_crossentropy_flatt


def train(dataset, model_name, learning_rate, weight_decay,
          num_epochs, max_patience, batch_size, optimizer='rmsprop',
          savepath='/Tmp/romerosa/deeppolyps/models/'):

    in_shape = (1, 512, 512)
    n_classes = 2

    # Build model
    print 'Building model'
    if model_name == 'fcn8':
        model = build_fcn8(in_shape,
                           regularize_weights=weight_decay,
                           nclasses=n_classes,
                           load_weights=False)
    else:
        raise ValueError('Unknown model')
    model.output

    # Compile model
    print 'Compiling model'
    model.compile(loss=categorical_crossentropy_flatt, optimizer="rmsprop")

    # Load data

    # Data augmentation
    # datagen = ImageDataGenerator(rotation_range=25,
    #                              shear_range=.41,
    #                              horizontal_flip=True,
    #                              vertical_flip=True,
    #                              fill_mode='reflect')


    # Early stopping and model saving
    early_stopping = EarlyStopping(monitor='val_loss', patience=50,
                                   verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath="./weights_ResUnet_finetune.hdf5",
                                   verbose=1,
                                   save_best_only=True)

    print('We are ready to run!')
    hist = model.fit_generator(datagen.flow(X_train,
                                            Y_train,
                                            batch_size=batch_size),
                               samples_per_epoch=2*len(X_train),
                               nb_epoch=num_epochs,
                               callbacks=[early_stopping, checkpointer],
                               validation_data=(X_val, Y_val))


def main():
    parser = argparse.ArgumentParser(description='Unet model training')
    parser.add_argument('-dataset',
                        default='camvid',
                        help='Dataset.')
    parser.add_argument('-model_name',
                        default='fcn8',
                        help='Model.')
    parser.add_argument('-learning_rate',
                        default=0.0001,
                        help='Learning Rate')
    parser.add_argument('-weight_decay',
                        default=0.0,
                        help='regularization constant')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=1000,
                        help='Optional. Int to indicate the max'
                        'number of epochs.')
    parser.add_argument('-max_patience',
                        type=int,
                        default=100,
                        help='Max patience (early stopping)')
    parser.add_argument('-batch_size',
                        type=int,
                        default=10,
                        help='Batch size')
    parser.add_argument('--optimizer',
                        '-opt',
                        default='rmsprop',
                        help='Optimizer')

    args = parser.parse_args()

    train(args.dataset, args.model_name, float(args.learning_rate),
          float(args.weight_decay), int(args.num_epochs),
          int(args.max_patience), int(args.batch_size),  args.optimizer)

if __name__ == "__main__":
    main()
