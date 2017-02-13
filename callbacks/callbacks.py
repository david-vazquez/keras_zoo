# Imports
from keras import backend as K
dim_ordering = K.image_dim_ordering()
from keras.callbacks import Callback, Progbar, ProgbarLogger
from keras.engine.training import GeneratorEnqueuer
from tools.save_images import save_img3
from tools.plot_history import plot_history
import numpy as np
import time

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
                 n_legend_rows=1, *args):
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

    def on_epoch_end(self, epoch, logs={}):

        # Create a data generator
        enqueuer = GeneratorEnqueuer(self.generator, pickle_safe=False)
        enqueuer.start(nb_worker=1, max_q_size=1, wait_time=0.05)

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
            #data = data_gen_queue.get()
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
                      self.color_map, self.classes, self.tag+str(_), self.void_label,
                      self.n_legend_rows)

        # Stop data generator
        if enqueuer is not None:
            enqueuer.stop()
