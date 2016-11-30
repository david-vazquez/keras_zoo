# Imports
from keras.callbacks import Callback, Progbar, ProgbarLogger
from keras.engine.training import generator_queue
from tools.save_images import save_img3, save_img4
import numpy as np
import seaborn as sns

# Install pydensecrf: https://github.com/lucasb-eyer/pydensecrf
# pip install git+https://github.com/lucasb-eyer/pydensecrf.git
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral,\
                             create_pairwise_gaussian, unary_from_softmax


# PROGBAR replacements
def progbar__set_params(self, params):
    self.params = params
    self.params['metrics'].extend(self.valid_metrics)


def progbar_on_epoch_begin(self, epoch, logs={}):
    if self.verbose:
        print('Epoch %d/%d' % (epoch + 1, self.nb_epoch))
        self.progbar = Progbar(target=self.params['nb_sample'],
                               verbose=self.verbose)
    self.seen = 0


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


# Conditional Random Field (CRF) iterative inference
def crf_inference(x, y, num_iter=5, bilateral=True, is01=False):
    # print ('X shape: ' + str(x.shape))
    # print ('Y shape: ' + str(y.shape))

    # Get shapes
    n_batches = y.shape[0]
    n_classes = y.shape[1]
    height = y.shape[2]
    width = y.shape[3]

    # Only implemented for mini batches of size 1
    if n_batches != 1:
        raise ValueError('Not implemented')

    # Create the DenseCRF2D object with the input size
    # width, height, nlabels TODO: Invert??
    d = dcrf.DenseCRF2D(height, width, n_classes)

    # Set the prediction as one row for each channel
    pred = y[0]
    pred = pred.reshape((n_classes, -1))
    # print ('Pred shape: ' + str(pred.shape))

    # Set the image as rows x columns x channels
    img = x[0]
    img = np.transpose(img, (1, 2, 0))
    # print ('Img shape: ' + str(img.shape))

    # Convert to numpy array
    img = np.array(img)
    # print ('Img shape: ' + str(img.shape))

    # Set to 0-255 range
    if is01:
        img = (255 * img).astype('uint8')
    else:
        img = img.astype('uint8')
        # print ('Max img: ' + str(np.max(img)))
        # print ('Min img: ' + str(np.min(img)))

    # Needed transformation
    img2 = np.zeros(img.shape).astype('uint8')
    img2 = img2 + img

    # set unary potentials (neg log probability)
    U = unary_from_softmax(pred)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the
    # locations only. Smoothness kernel.
    # sxy: gaussian x, y std
    # compat: ways to weight contributions, a number for potts compatibility,
    #     vector for diagonal compatibility, an array for matrix compatibility
    # kernel: kernel used, CONST_KERNEL, FULL_KERNEL, DIAG_KERNEL
    # normalization: NORMALIZE_AFTER, NORMALIZE_BEFORE,
    #     NO_NORMALIZAITION, NORMALIZE_SYMMETRIC
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Appearance kernel. This adds the color-dependent term, i.e. features
    # are (x,y,r,g,b). im is an image-array, e.g. im.dtype == np.uint8
    # and im.shape == (640,480,3) to set sxy and srgb perform grid search
    # on validation set
    if bilateral:
        d.addPairwiseBilateral(sxy=(3, 3), srgb=(13, 13, 13),
                               rgbim=img2, compat=10, kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Perform the inference
    Q = d.inference(num_iter)

    # Reshape output to n_classes x height x width
    Q = np.reshape(Q, (n_classes, height, width))
    # print ('Out shape: ' + str(Q.shape))

    # Add batch dimension
    output = np.expand_dims(Q, axis=0)
    # print ('Out shape: ' + str(output.shape))

    return output


# Computes the desired metrics (And saves images)
def compute_metrics(model, val_gen, epoch_length, nclasses, metrics,
                    color_map, tag, void_label, out_images_folder, epoch,
                    save_all_images=False, useCRF=False):

    if 'test_jaccard_perclass' in metrics:
        metrics.remove('test_jaccard_perclass')
        for i in range(nclasses):
            metrics.append(str(i) + '_test_jacc_percl')

    # Create a data generator
    data_gen_queue, _stop = generator_queue(val_gen, max_q_size=10)

    # Create the metrics output dictionary
    metrics_out = {}

    # Create the confusion matrix
    cm = np.zeros((nclasses, nclasses))
    hist_loss = []

    # Process the dataset
    for _ in range(epoch_length):
        if useCRF:
            print(str(_) + ' from ' + str(epoch_length))

        # Get data for this minibatch
        data = data_gen_queue.get()
        x_true = data[0]
        y_true = data[1].astype('int32')

        # Get prediction for this minibatch
        y_pred = model.predict(x_true)

        # CRF
        if useCRF:
            # Add void channel
            void_channel = np.zeros((1, 1, y_pred.shape[2], y_pred.shape[3]))
            void_channel[(y_true == void_label).nonzero()] = 1
            y_pred = np.concatenate((y_pred, void_channel), axis=1)

            # Adjust probabilities of void pixels (1 for void and 0 other)
            mask = np.ones((1, 1, y_pred.shape[2], y_pred.shape[3]))
            mask[(y_true == void_label).nonzero()] = 0
            for i in xrange(nclasses):
                y_pred[0, i, :, :] = y_pred[0, i, :, :]*mask[0]
            # print ('Y_pred shape: ' + str(y_pred.shape))

            # Make CRF inference
            y_pred_crf = crf_inference(x_true, y_pred, num_iter=5,
                                       bilateral=False)

            # Compute the loss
            hist_loss.append(cat_cross_entropy_voids(y_pred_crf, y_true,
                                                     void_label))

            # Compute the argmax
            y_pred = np.argmax(y_pred, axis=1)
            y_pred_crf = np.argmax(y_pred_crf, axis=1)
        else:
            # Compute the loss
            hist_loss.append(cat_cross_entropy_voids(y_pred, y_true,
                                                     void_label))

            # Compute the argmax
            y_pred = np.argmax(y_pred, axis=1)

        # Reshape y_true
        y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[2],
                                     y_true.shape[3]))

        # Save output images (Only first minibatch)
        if save_all_images or not save_all_images and _ == 0:

            if useCRF:
                save_img4(x_true, y_true, y_pred, y_pred_crf,
                          out_images_folder, epoch,
                          color_map, tag+str(_), void_label)
            else:
                save_img3(x_true, y_true, y_pred, out_images_folder, epoch,
                          color_map, tag+str(_), void_label)

        # Make y_true and y_pred flatten
        if useCRF:
            y_pred = y_pred_crf.flatten()
            y_true = y_true.flatten()
        else:
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()

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
        elif m.endswith('jacc_percl'):
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
                 valid_gen, valid_epoch_length, valid_metrics,
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

        self.test_gen = test_gen
        self.test_epoch_length = test_epoch_length
        self.test_metrics = test_metrics

        # Create the colormaping for showing labels
        #self.color_map = sns.hls_palette(n_classes+1)
        self.color_map = [
            (255/255., 0, 0),                   # Background
            (192/255., 192/255., 128/255.),     # Polyp
            (128/255., 64/255., 128/255.),      # Lumen
            (0, 0, 255/255.),                   # Specularity
            (0, 255/255., 0),         #
            (192/255., 128/255., 128/255.),     #
            (64/255., 64/255., 128/255.),       #
        ]
        self.last_epoch = 0
        if 'val_jaccard_perclass' in self.valid_metrics:
            self.valid_metrics.remove('val_jaccard_perclass')
            for i in range(n_classes):
                self.valid_metrics.append(str(i) + '_val_jacc_percl')
        setattr(ProgbarLogger, 'valid_metrics',
                self.valid_metrics)
        setattr(ProgbarLogger, '_set_params', progbar__set_params)
        # setattr(ProgbarLogger, 'on_epoch_begin', progbar_on_epoch_begin)

    # Compute metrics for validation set at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
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
        for k, v in metrics_out.iteritems():
            logs[k] = v
