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


def my_label2rgb(labels, colors, bglabel=None, bg_color=(0., 0., 0.)):
    output = np.zeros(labels.shape + (3,), dtype=np.float64)
    for i in range(len(colors)):
        if i != bglabel:
            output[(labels == i).nonzero()] = colors[i]
    if bglabel is not None:
        output[(labels == bglabel).nonzero()] = bg_color
    return output


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


# Computes the confusion matrix
def confusion_matrix(model, val_gen, epoch_length, nclasses,
                     color_map, tag, void_label, out_images_folder, epoch):
    # Create a data generator
    data_gen_queue, _stop = generator_queue(val_gen, max_q_size=10)

    # Create the confusion matrix
    CM = np.zeros((nclasses, nclasses))
    for _ in range(epoch_length):
        # Get data for this minibatch
        data = data_gen_queue.get()
        x_true = data[0]
        y_true = data[1]

        # Get prediction for this minibatch
        y_pred = model.predict(x_true)
        y_pred = np.argmax(y_pred, axis=1)

        # Show shapes
        # print ('x_true shape: ' + str(x_true.shape))
        # print ('y_true shape: ' + str(y_true.shape))
        # print ('y_pred shape: ' + str(y_pred.shape))

        # Save output images
        if _ == 0:
            y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[2],
                                         y_true.shape[3]))
            save_img(x_true, y_true, y_pred, out_images_folder, epoch,
                     color_map, tag, void_label)

        # Make y_true and y_pred flatten
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Show flatten shapes
        # print ('y_true shape: ' + str(y_true.shape))
        # print ('y_pred shape: ' + str(y_pred.shape))

        # Fill the confusion matrix
        for i in range(nclasses):
            for j in range(nclasses):
                CM[i, j] += ((y_pred == i) * (y_true == j)).sum()

    # Stop data generator
    _stop.set()
    # Return confusion matrix
    return CM


# Jaccard value computation callback
class ValJaccard(Callback):
    # Constructor
    def __init__(self, nclasses, valid_gen, epoch_length, void_label, *args):
        super(Callback, self).__init__()
        self.nclasses = nclasses
        self.valid_gen = valid_gen
        self.epoch_length = epoch_length
        self.void_label = void_label
        # Create the colormaping for showing labels
        self.color_map = sns.hls_palette(nclasses+1)

    # Compute jaccard value at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Compute the confusion matrix
        CM = confusion_matrix(self.model, self.valid_gen, self.epoch_length,
                              self.nclasses, color_map=self.color_map,
                              tag="valid", void_label=self.void_label,
                              out_images_folder='./', epoch=epoch)

        # Compute and print the TP per class
        TP_perclass = CM.diagonal().astype('float32')
        print(TP_perclass.astype('int32'))
        # Compute the jaccard
        jaccard = TP_perclass/(CM.sum(1) + CM.sum(0) - TP_perclass)
        print(CM.sum(1).astype('int32'))
        print(CM.sum(0).astype('int32'))
        # Compute jaccard mean ignoring NaNs
        jaccard = np.nanmean(jaccard)
        print('Jaccard: ' + str(jaccard))
        print('Acc: ' + str(TP_perclass.sum()/CM.sum()))
