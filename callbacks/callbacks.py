# Imports
from keras.callbacks import Callback


# Computes the confusion matrix
def confusion_matrix(model, val_gen, epoch_length, nclasses=21):
    # Create a data generator
    data_gen_queue, _stop = generator_queue(val_gen, max_q_size=10)

    # Create the confusion matrix
    CM = np.zeros((nclasses, nclasses))
    loss = 0
    for _ in range(epoch_length):
        # Get data for this minibatch
        data = data_gen_queue.get()
        # Get prediction for this minibatch
        y_pred = model.predict(data[0])
        # Get y_true and make it flat
        y_true = data[1]
        y_true = y_true.flatten()
        # Reshape y_pred (b01,c)
        sh = y_pred.shape
        y_pred = np.reshape(y_pred, (sh[0] * sh[1] * sh[2], sh[3]))
        # Get argmax of the prediction
        y_pred = np.argmax(y_pred, axis=-1)
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
    def __init__(self, nclasses, valid_gen, epoch_length, *args):
        super(Callback, self).__init__()
        self.nclasses = nclasses
        self.valid_gen = valid_gen
        self.epoch_length = epoch_length

    # Compute jaccard value at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Compute the confusion matrix
        CM = confusion_matrix(self.model, self.valid_gen, self.epoch_length)
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

# class Save_images(Callback):
#     def __init__(self,
#                  path='./'):
#         super(Save_images, self).__init__()
#         self.path = path
#
#     def on_epoch_end(self, epoch, logs={}):
