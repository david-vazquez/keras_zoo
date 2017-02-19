import math
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from callbacks import (History_plot, Jacc_new, Save_results)


# Create callbacks
class Callbacks_Factory():
    def __init__(self):
        pass

    def make(self, cf, valid_gen):
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
                                save_path=cf.savepath,
                                generator=valid_gen,
                                epoch_length=int(math.ceil(cf.save_results_nsamples/float(cf.save_results_batch_size))),
                                color_map=cf.dataset.color_map,
                                classes=cf.dataset.classes,
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

        # Save the log
        cb += [CSVLogger(os.path.join(cf.savepath, 'logFile.csv'),
                         separator=',', append=False)]

        # Output the list of callbacks
        return cb
