import math
import os

from keras.callbacks import (EarlyStopping, ModelCheckpoint, CSVLogger,
                             LearningRateScheduler, TensorBoard)

from callbacks import (History_plot, Jacc_new, Save_results,
                       LearningRateSchedulerBatch, Scheduler)


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

        # Learning rate scheduler
        if cf.LRScheduler_enabled:
            print('   Learning rate cheduler by batch')
            scheduler = Scheduler(cf.LRScheduler_type, cf.learning_rate,
                                  cf.LRScheduler_M, cf.LRScheduler_decay,
                                  cf.LRScheduler_S, cf.LRScheduler_power)

            if cf.LRScheduler_batch_epoch == 'batch':
                cb += [LearningRateSchedulerBatch(scheduler.scheduler_function)]
            elif cf.LRScheduler_batch_epoch == 'epoch':
                cb += [LearningRateScheduler(scheduler.scheduler_function)]
            else:
                raise ValueError('Unknown scheduler mode: ' + LRScheduler_batch_epoch)

        # TensorBoard callback
        if cf.TensorBoard_enabled:
            print('   Tensorboard')
            if cf.TensorBoard_logs_folder is None:
                log_dir = os.path.join(cf.usr_path, 'TensorBoardLogs')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            cb += [TensorBoard(log_dir=log_dir,
                               histogram_freq=cf.TensorBoard_histogram_freq,
                               write_graph=cf.TensorBoard_write_graph,
                               write_images=cf.TensorBoard_write_images)]

        # Output the list of callbacks
        return cb
