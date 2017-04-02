# Imports
from keras import backend as K
dim_ordering = K.image_dim_ordering()
from keras.callbacks import (Callback, Progbar, ProgbarLogger,
                             LearningRateScheduler)
from keras.engine.training import GeneratorEnqueuer
from tools.save_images import save_img3
from tools.plot_history import plot_history
import numpy as np
import time
import math

# PROGBAR replacements
def progbar__set_params(self, params):
    self.params = params
    print('Anado metrics!!!!!!!!: ' + str(self.add_metrics))
    self.params['metrics'].extend(self.add_metrics)


def progbar_on_epoch_begin(self, epoch, logs={}):
    if self.verbose:
        print('Epoch %d/%d' % (epoch + 1, self.nb_epoch))
        self.progbar = Progbar(target=self.params['nb_sample'],
                               verbose=self.verbose)
    # self.params['metrics'].extend(self.add_metrics)
    self.seen = 0


def progbar_on_batch_end(self, batch, logs={}):
    batch_size = logs.get('size', 0)
    self.seen += batch_size

    for k in self.params['metrics']:
        if k in logs and k not in self.remove_metrics:
            self.log_values.append((k, logs[k]))

    for k in self.add_metrics:
        if k in logs and k not in self.remove_metrics:
            self.log_values.append((k, logs[k]))

    # skip progbar update for the last batch;
    # will be handled by on_epoch_end
    if self.verbose and self.seen < self.params['nb_sample']:
        self.progbar.update(self.seen, self.log_values)


def progbar_on_epoch_end(self, epoch, logs={}):
    for k in self.params['metrics']:
        if k in logs and k not in self.remove_metrics:
            self.log_values.append((k, logs[k]))

    for k in self.add_metrics:
        if k in logs and k not in self.remove_metrics:
            self.log_values.append((k, logs[k]))

    if self.verbose:
        self.progbar.update(self.seen, self.log_values, force=True)


# Plot history
class History_plot(Callback):

    # Constructor
    def __init__(self, n_classes, savepath, train_metrics, valid_metrics,
                 best_metric, best_type, verbose=False, *args):
        super(Callback, self).__init__()
        # Save input parameters
        self.n_classes = n_classes
        self.savepath = savepath
        self.verbose = verbose
        self.train_metrics = train_metrics
        self.valid_metrics = valid_metrics
        self.best_metric = best_metric
        self.best_type = best_type

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        plot_history(self.history, self.savepath, self.n_classes,
                     train_metrics=self.train_metrics,
                     valid_metrics=self.valid_metrics,
                     best_metric=self.best_metric,
                     best_type=self.best_type,
                     verbose=self.verbose)


# Compute the jaccard value
class Jacc_new(Callback):

    # Constructor
    def __init__(self, n_classes, *args):
        super(Callback, self).__init__()
        # Save input parameters
        self.n_classes = n_classes
        self.I = np.zeros(self.n_classes)
        self.U = np.zeros(self.n_classes)
        self.jacc_percl = np.zeros(self.n_classes)
        self.val_I = np.zeros(self.n_classes)
        self.val_U = np.zeros(self.n_classes)
        self.val_jacc_percl = np.zeros(self.n_classes)

        self.remove_metrics = []
        for i in range(n_classes):
            self.remove_metrics.append('I' + str(i))
            self.remove_metrics.append('U' + str(i))
            self.remove_metrics.append('val_I' + str(i))
            self.remove_metrics.append('val_U' + str(i))

        self.add_metrics = []
        self.add_metrics.append('jaccard')
        self.add_metrics.append('val_jaccard')
        for i in range(n_classes):
            self.add_metrics.append(str(i) + '_jacc')
            self.add_metrics.append(str(i) + '_val_jacc')
        setattr(ProgbarLogger, 'add_metrics', self.add_metrics)
        setattr(ProgbarLogger, 'remove_metrics', self.remove_metrics)
        setattr(ProgbarLogger, '_set_params', progbar__set_params)
        setattr(ProgbarLogger, 'on_batch_end', progbar_on_batch_end)
        setattr(ProgbarLogger, 'on_epoch_end', progbar_on_epoch_end)

    def on_batch_end(self, batch, logs={}):
        for i in range(self.n_classes):
            self.I[i] = logs['I'+str(i)]
            self.U[i] = logs['U'+str(i)]
            self.jacc_percl[i] = self.I[i] / self.U[i]
            # logs[str(i)+'_jacc'] = self.jacc_percl[i]
        self.jacc_percl = self.I / self.U
        self.jacc = np.nanmean(self.jacc_percl)
        logs['jaccard'] = self.jacc

    def on_epoch_end(self, epoch, logs={}):
        for i in range(self.n_classes):
            self.I[i] = logs['I'+str(i)]
            self.U[i] = logs['U'+str(i)]
            self.jacc_percl[i] = self.I[i] / self.U[i]
            logs[str(i)+'_jacc'] = self.jacc_percl[i]
        self.jacc = np.nanmean(self.jacc_percl)
        logs['jaccard'] = self.jacc

        for i in range(self.n_classes):
            self.val_I[i] = logs['val_I'+str(i)]
            self.val_U[i] = logs['val_U'+str(i)]
            self.val_jacc_percl[i] = self.val_I[i] / self.val_U[i]
            logs[str(i)+'_val_jacc'] = self.val_jacc_percl[i]
        self.val_jacc = np.nanmean(self.val_jacc_percl)
        logs['val_jaccard'] = self.val_jacc


# Save the image results
class Save_results(Callback):
    def __init__(self, n_classes, void_label, save_path,
                 generator, epoch_length, color_map, classes, tag,
                 n_legend_rows=1, nb_worker=5, max_q_size=10, *args):
        super(Callback, self).__init__()
        self.n_classes = n_classes
        self.void_label = void_label
        self.save_path = save_path
        self.generator = generator
        self.epoch_length = epoch_length
        self.color_map = color_map
        self.classes = classes
        self.n_legend_rows = n_legend_rows
        self.tag = tag
        self.nb_worker = nb_worker
        self.max_q_size = max_q_size

    def on_epoch_end(self, epoch, logs={}):

        # Create a data generator
        enqueuer = GeneratorEnqueuer(self.generator, pickle_safe=True)
        enqueuer.start(nb_worker=self.nb_worker, max_q_size=self.max_q_size,
                       wait_time=0.05)

        # Process the dataset
        for _ in range(self.epoch_length):

            # Get data for this minibatch
            data = None
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    data = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.05)
            x_true = data[0]
            y_true = data[1].astype('int32')

            # Get prediction for this minibatch
            y_pred = self.model.predict(x_true)

            # Reshape y_true and compute the y_pred argmax
            if K.image_dim_ordering() == 'th':
                y_pred = np.argmax(y_pred, axis=1)
                y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[2],
                                             y_true.shape[3]))
            else:
                y_pred = np.argmax(y_pred, axis=3)
                y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                             y_true.shape[2]))
            # Save output images
            save_img3(x_true, y_true, y_pred, self.save_path, epoch,
                      self.color_map, self.classes, self.tag+str(_),
                      self.void_label, self.n_legend_rows)

        # Stop data generator
        if enqueuer is not None:
            enqueuer.stop()


class Scheduler():
    """ Learning rate scheduler function
    # Arguments
        scheduler_type: ['linear' | 'step' | 'square' | 'sqrt']
        lr: initial learning rate
        M: number of learning iterations
        decay: decay coefficient
        S: step iteration
        from: https://arxiv.org/pdf/1606.02228.pdf
        poly from: https://arxiv.org/pdf/1606.00915.pdf
    """
    def __init__(self, scheduler_type='linear', lr=0.001, M=320000,
                 decay=0.1, S=100000, power=0.9):
        # Save parameters
        self.scheduler_type = scheduler_type
        self.lr = float(lr)
        self.decay = float(decay)
        self.M = float(M)
        self.S = S
        self.power = power

        # Get function
        if self.scheduler_type == 'linear':
            self.scheduler_function = self.linear_scheduler
        elif self.scheduler_type == 'step':
            self.scheduler_function = self.step_scheduler
        elif self.scheduler_type == 'square':
            self.scheduler_function = self.square_scheduler
        elif self.scheduler_type == 'sqrt':
            self.scheduler_function = self.sqrt_scheduler
        elif self.scheduler_type == 'poly':
            self.scheduler_function = self.poly_scheduler
        else:
            raise ValueError('Unknown scheduler: ' + self.scheduler_type)

    def step_scheduler(self, i):
        return self.lr * math.pow(self.decay, math.floor(i/self.M))

    def linear_scheduler(self, i):
        return self.lr * (1. - i/self.M)

    def square_scheduler(self, i):
        return self.lr * ((1. - i/self.M)**2)

    def sqrt_scheduler(self, i):
        return self.lr * math.sqrt(1. - i/self.M)

    def poly_scheduler(self, i):
        return self.lr * ((1. - i/self.M)**self.power)


class LearningRateSchedulerBatch(Callback):
    """Learning rate scheduler.

    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    """

    def __init__(self, schedule):
        super(LearningRateSchedulerBatch, self).__init__()
        self.schedule = schedule
        self.iter = 0

    def on_batch_begin(self, batch, logs=None):
        self.iter += 1
        self.change_lr(self.iter)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.schedule(self.iter)
        print('   New lr: ' + str(lr))

    def change_lr(self, iteration):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.schedule(iteration)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
