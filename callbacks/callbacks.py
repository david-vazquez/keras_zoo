# Imports
from keras.callbacks import Callback, Progbar, ProgbarLogger
from keras.engine.training import generator_queue
from tools.save_images import save_img3
from tools.plot_history import plot_history
import numpy as np
import time

# PROGBAR replacements
def progbar__set_params(self, params):
    self.params = params
    self.params['metrics'].extend(self.add_metrics)


def progbar_on_epoch_begin(self, epoch, logs={}):
    if self.verbose:
        print('Epoch %d/%d' % (epoch + 1, self.nb_epoch))
        self.progbar = Progbar(target=self.params['nb_sample'],
                               verbose=self.verbose)
    self.seen = 0


def progbar_on_batch_end(self, batch, logs={}):
    batch_size = logs.get('size', 0)
    self.seen += batch_size

    for k in self.params['metrics']:
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
    if self.verbose:
        self.progbar.update(self.seen, self.log_values, force=True)


# Compute the masked categorical crossentropy
def cat_cross_entropy_voids(y_pred, y_true, void_label, _EPS=10e-8,
                            dim_ordering='th'):

    # Move classes to the end (bc01 -> b01c)
    if dim_ordering == 'th':
        y_true = y_true.transpose([0, 2, 3, 1])
        y_pred = y_pred.transpose([0, 2, 3, 1])

    # Reshape to (b01, n_classes)
    y_true = y_true.flatten()  # reshape(-1, sh_t[-1])
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    # Compute the void mask
    mask = np.ones_like(y_true).astype('int32')
    if void_label is not None:
        mask[y_true == void_label] = 0.
        y_true[y_true == void_label] = 0.

    # Avoid numerical instability with _EPSILON clipping
    y_pred = np.clip(y_pred, _EPS, 1.0 - _EPS)

    # Compute the negative log likelihood for each pixel
    cost = np.zeros_like(y_pred[:, 0])
    for i in range(len(cost)):
        cost[i] = -np.log(y_pred[i, y_true[i]])

    # Compute the average cost
    cost *= mask
    cost = np.sum(cost) / np.sum(mask).astype(float)

    # Return the cost
    return cost


# Computes the desired metrics (And saves images)
def compute_metrics(model, val_gen, epoch_length, nclasses, metrics,
                    color_map, tag, void_label, out_images_folder, epoch,
                    save_all_images=False, useCRF=False):

    if 'test_jaccard_perclass' in metrics:
        metrics.remove('test_jaccard_perclass')
        for i in range(nclasses):
            metrics.append(str(i) + '_test_jacc')

    # Create a data generator
    data_gen_queue, _stop, _generator_threads = generator_queue(val_gen,
                                                                max_q_size=10)

    # Create the metrics output dictionary
    metrics_out = {}

    # Create the confusion matrix
    cm = np.zeros((nclasses, nclasses))
    hist_loss = []

    load_data_time = 0
    predict_time = 0
    loss_time = 0
    argmax_time = 0
    cm_time = 0

    # Process the dataset
    for _ in range(epoch_length):
        # if useCRF:
        #     print(str(_) + ' from ' + str(epoch_length))

        # Get data for this minibatch
        start_time = time.time()
        data = data_gen_queue.get()
        x_true = data[0]
        y_true = data[1].astype('int32')
        print(x_true.shape)
        load_data_time += time.time() - start_time

        # Get prediction for this minibatch
        start_time = time.time()
        y_pred = model.predict(x_true)
        predict_time += time.time() - start_time

        # Compute the loss
        start_time = time.time()
        hist_loss.append(cat_cross_entropy_voids(y_pred, y_true,
                                                 void_label))
        loss_time += time.time() - start_time

        # Compute the argmax
        start_time = time.time()
        y_pred = np.argmax(y_pred, axis=1)
        argmax_time += time.time() - start_time

        # Reshape y_true
        y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[2],
                                     y_true.shape[3]))

        # Save output images (Only first minibatch)
        if save_all_images or not save_all_images and _ == 0:
            save_img3(x_true, y_true, y_pred, out_images_folder, epoch,
                      color_map, tag+str(_), void_label)

        # Make y_true and y_pred flatten
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        # Fill the confusion matrix
        start_time = time.time()
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] += ((y_pred == i) * (y_true == j)).sum()
        cm_time += time.time() - start_time

    # Stop data generator
    _stop.set()

    print('load_data_time: ' + str(load_data_time))
    print('predict_time: ' + str(predict_time))
    print('loss_time: ' + str(loss_time))
    print('argmax_time: ' + str(argmax_time))
    print('cm_time: ' + str(cm_time))

    # Compute metrics
    TP_perclass = cm.diagonal().astype('float32')
    jaccard_perclass = TP_perclass/(cm.sum(1) + cm.sum(0) - TP_perclass)
    jaccard = np.nanmean(jaccard_perclass)
    accuracy = TP_perclass.sum()/cm.sum()
    loss = np.mean(hist_loss)

    # Fill metrics output
    for m in metrics:
        if m.endswith('jaccard'):
            metrics_out[m] = jaccard
        elif m.endswith('jacc'):
            metrics_out[m] = jaccard_perclass[int(m.split('_')[0])]
        elif m.endswith('acc') or m.endswith('accuracy'):
            metrics_out[m] = accuracy
        elif m.endswith('cm'):
            metrics_out[m] = cm
        elif m.endswith('loss'):
            metrics_out[m] = loss
        else:
            print('Metric {} unknown'.format(m))

    # Return metrics out
    return metrics_out



# Jaccard value computation callback
class Evaluate_model(Callback):
    # Constructor
    def __init__(self, n_classes, void_label, save_path,
                 valid_gen, valid_epoch_length, valid_metrics, color_map,
                 test_gen=None, test_epoch_length=None, test_metrics=None,
                 *args):
        super(Callback, self).__init__()
        # Save input parameters
        self.n_classes = n_classes
        self.void_label = void_label
        self.save_path = save_path

        self.valid_gen = valid_gen
        self.valid_epoch_length = valid_epoch_length
        self.valid_metrics = valid_metrics
        self.color_map = color_map

        self.test_gen = test_gen
        self.test_epoch_length = test_epoch_length
        self.test_metrics = test_metrics

        self.last_epoch = 0
        self.remove_metrics = []

        if 'val_jaccard_perclass' in self.valid_metrics:
            self.valid_metrics.remove('val_jaccard_perclass')
            for i in range(n_classes):
                self.valid_metrics.append(str(i) + '_val_jacc')

        setattr(ProgbarLogger, 'add_metrics', self.valid_metrics)
        setattr(ProgbarLogger, 'remove_metrics', self.remove_metrics)
        setattr(ProgbarLogger, '_set_params', progbar__set_params)
        setattr(ProgbarLogger, 'on_batch_end', progbar_on_batch_end)
        setattr(ProgbarLogger, 'on_epoch_end', progbar_on_epoch_end)
        # setattr(ProgbarLogger, 'on_epoch_begin', progbar_on_epoch_begin)

    # Compute metrics for validation set at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        start_time = time.time()
        # Compute the metrics
        metrics_out = compute_metrics(self.model, self.valid_gen,
                                      self.valid_epoch_length, self.n_classes,
                                      metrics=self.valid_metrics,
                                      color_map=self.color_map, tag="valid",
                                      void_label=self.void_label,
                                      out_images_folder=self.save_path,
                                      epoch=epoch)
        self.last_epoch = epoch

        # Save the metrics in the logs
        for k, v in metrics_out.items():
            logs[k] = v
        print ('Validation time: ' + str(time.time()-start_time))


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
        self.jacc = np.mean(self.jacc_percl)
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
                 generator, epoch_length, color_map, tag,
                 *args):
        super(Callback, self).__init__()
        self.n_classes = n_classes
        self.void_label = void_label
        self.save_path = save_path
        self.generator = generator
        self.epoch_length = epoch_length
        self.color_map = color_map
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        # Create a data generator
        data_gen_queue, _stop, _generator_threads = generator_queue(self.generator,
                                                                    max_q_size=1)

        # Process the dataset
        for _ in range(self.epoch_length):

            # Get data for this minibatch
            data = data_gen_queue.get()
            x_true = data[0]
            y_true = data[1].astype('int32')

            # Get prediction for this minibatch
            y_pred = self.model.predict(x_true)

            # Compute the argmax
            y_pred = np.argmax(y_pred, axis=1)

            # Reshape y_true
            y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[2],
                                         y_true.shape[3]))

            # Save output images
            save_img3(x_true, y_true, y_pred, self.save_path, epoch,
                      self.color_map, self.tag+str(_), self.void_label)

        # Stop data generator
        _stop.set()
