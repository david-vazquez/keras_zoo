# Imports
from keras.callbacks import Callback
from keras.engine.training import generator_queue
import numpy as np
from keras import backend as K
import theano.tensor as T

from skimage.color import label2rgb
from skimage import img_as_float
from skimage.color import rgb2gray, gray2rgb
import scipy.misc
import seaborn as sns


# Converts a label mask to RGB to be shown
def my_label2rgb(labels, colors, bglabel=None, bg_color=(0., 0., 0.)):
    output = np.zeros(labels.shape + (3,), dtype=np.float64)
    for i in range(len(colors)):
        if i != bglabel:
            output[(labels == i).nonzero()] = colors[i]
    if bglabel is not None:
        output[(labels == bglabel).nonzero()] = bg_color
    return output


# Converts a label mask to RGB to be shown and overlaps over an image
def my_label2rgboverlay(labels, colors, image, bglabel=None,
                        bg_color=(0., 0., 0.), alpha=0.2):
    image_float = gray2rgb(img_as_float(rgb2gray(image)))
    label_image = my_label2rgb(labels, colors, bglabel=bglabel,
                               bg_color=bg_color)
    output = image_float * alpha + label_image * (1 - alpha)
    return output


# Save images
def save_img(image_batch, mask_batch, output, out_images_folder, epoch,
             color_map, tag, void_label):
    output[(mask_batch == void_label).nonzero()] = void_label
    images = []
    for j in xrange(output.shape[0]):
        img = image_batch[j].transpose((1, 2, 0)) / 255.
        label_out = my_label2rgb(output[j], bglabel=void_label,
                                 colors=color_map)
        label_mask = my_label2rgboverlay(mask_batch[j], colors=color_map,
                                         image=img, bglabel=void_label,
                                         alpha=0.2)
        label_overlay = my_label2rgboverlay(output[j], colors=color_map,
                                            image=img, bglabel=void_label,
                                            alpha=0.5)

        combined_image = np.concatenate((img, label_mask, label_out,
                                         label_overlay), axis=1)
        out_name = out_images_folder + tag + '_epoch' + str(epoch) + '_img' + str(j) + '.png'
        scipy.misc.toimage(combined_image).save(out_name)
        images.append(combined_image)
    return images


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
    for i in xrange(len(cost)):
        cost[i] = -np.log(y_pred[i, y_true[i]])

    # Compute the average cost
    cost *= mask
    cost = np.sum(cost) / np.sum(mask).astype(float)

    # Return the cost
    return cost


# Computes the desired metrics (And saves images)
def compute_metrics(model, val_gen, epoch_length, nclasses, metrics,
                    color_map, tag, void_label, out_images_folder, epoch):
    # Create a data generator
    data_gen_queue, _stop = generator_queue(val_gen, max_q_size=10)

    # Create the metrics output dictionary
    metrics_out = {}

    # Create the confusion matrix
    cm = np.zeros((nclasses, nclasses))
    hist_loss = []

    # Process the dataset
    for _ in range(epoch_length):
        # Get data for this minibatch
        data = data_gen_queue.get()
        x_true = data[0]
        y_true = data[1].astype('int32')

        # Get prediction for this minibatch
        y_pred = model.predict(x_true)

        # Compute the loss
        hist_loss.append(cat_cross_entropy_voids(y_pred, y_true, void_label))

        # Compute the argmax
        y_pred = np.argmax(y_pred, axis=1)

        # Save output images (Only first minibatch)
        if _ == 0:
            y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[2],
                                         y_true.shape[3]))
            save_img(x_true, y_true, y_pred, out_images_folder, epoch,
                     color_map, tag, void_label)

        # Make y_true and y_pred flatten
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Fill the confusion matrix
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] += ((y_pred == i) * (y_true == j)).sum()

    # Stop data generator
    _stop.set()

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
        elif m.endswith('jaccard_perclass'):
            metrics_out[m] = jaccard_perclass
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
class ValJaccard(Callback):
    # Constructor
    def __init__(self, nclasses, valid_gen, epoch_length, void_label,
                 out_images_folder, metrics, *args):
        super(Callback, self).__init__()
        # Save input parameters
        self.nclasses = nclasses
        self.valid_gen = valid_gen
        self.epoch_length = epoch_length
        self.void_label = void_label
        self.out_images_folder = out_images_folder
        self.metrics = metrics

        # Create the colormaping for showing labels
        self.color_map = sns.hls_palette(nclasses+1)

    # Compute jaccard value at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Compute the metrics
        metrics_out = compute_metrics(self.model, self.valid_gen,
                                      self.epoch_length, self.nclasses,
                                      metrics=self.metrics,
                                      color_map=self.color_map, tag="valid",
                                      void_label=self.void_label,
                                      out_images_folder=self.out_images_folder,
                                      epoch=epoch)

        # Save the metrics in the logs
        for k, v in metrics_out.iteritems():
            logs[k] = v
        print ('logs: ' + str(logs))
