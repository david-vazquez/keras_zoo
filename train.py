import argparse

from keras.callbacks import (EarlyStopping,
                             ModelCheckpoint)
from keras.optimizers import RMSprop
from fcn8 import build_fcn8
from loader_sem_seg import ImageDataGenerator
from metrics import categorical_crossentropy_flatt


def train(dataset, model_name, learning_rate, weight_decay,
          num_epochs, max_patience, batch_size, optimizer='rmsprop',
          savepath='/home/michal/tmp/',
          train_path='/home/michal/polyps/CVC-612/',
          val_path='/home/michal/polyps/CVC-300/'):

    crop_size = (224, 224)
    in_shape = (3, None, None)
    n_classes = 5

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
    optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-8, clipnorm=10)
    model.compile(loss=categorical_crossentropy_flatt, optimizer=optimizer)
    model.summary()

    # Data augmentation
    dg_tr = ImageDataGenerator(crop_size=crop_size)
    dg_ts = ImageDataGenerator()

    # Load data
    train_generator = dg_tr.flow_from_directory(train_path + 'bbdd',
                                                batch_size=10,
                                                gt_directory=train_path + 'labelled',
                                                target_size=crop_size,
                                                class_mode='seg_map')

    val_generator = dg_ts.flow_from_directory(val_path + 'bbdd',
                                              batch_size=4,
                                              gt_directory=val_path + 'labelled',
                                              target_size=(500, 574),
                                              class_mode='seg_map')

    # Early stopping and model saving
    early_stopping = EarlyStopping(monitor='val_loss', patience=50,
                                   verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath="./weights.hdf5",
                                   verbose=1,
                                   save_best_only=True)

    print('We are ready to run!')
    hist = model.fit_generator(train_generator,
                               samples_per_epoch=600,
                               nb_epoch=2,
                               nb_worker=5,
                               max_q_size=50)#,
                               # callbacks=[early_stopping, checkpointer],
                               # validation_data=val_generator,
                               # nb_val_samples=300)


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
          int(args.max_patience), int(args.batch_size),  args.optimizer,
          savepath='/home/michal/tmp',
          train_path='/home/michal/polyps/CVC-612/',
          val_path='/home/michal/polyps/CVC-300/')

if __name__ == "__main__":
    main()



