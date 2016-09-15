# Imports
import matplotlib.pyplot as plt
import numpy as np


# Plot training hisory
def plot_history(hist, save_path):
    # Compute the best values to add to the legend
    best_index = np.argmax(hist.history['val_jaccard'])
    b_tr_loss = np.min(hist.history['loss'])
    b_tr_loss2 = hist.history['loss'][best_index]
    b_va_loss = np.min(hist.history['val_loss'])
    b_va_loss2 = hist.history['val_loss'][best_index]
    b_va_jaccard = np.max(hist.history['val_jaccard'])
    b_va_acc = np.max(hist.history['val_acc'])
    b_va_acc2 = hist.history['val_acc'][best_index]

    print ('Best epoch: {}. Train loss: {}. Val loss: {}. Val accuracy:{}. Val Jaccard: {}.'.format(best_index, b_tr_loss2, b_va_loss2, b_va_acc2, b_va_jaccard))

    # Plot the metrics
    try:
        plt.plot(hist.history['loss'],
                 label='train loss ({:.3f},{:.3f})'.format(b_tr_loss, b_tr_loss2))
        plt.plot(hist.history['val_loss'],
                 label='valid loss ({:.3f},{:.3f})'.format(b_va_loss, b_va_loss2))
        plt.plot(hist.history['val_acc'],
                 label='valid acc ({:.3f},{:.3f})'.format(b_va_acc, b_va_acc2))
        plt.plot(hist.history['val_jaccard'],
                 label='valid jaccard ({:.3f})'.format(b_va_jaccard))

        # Add title
        plt.title('Model training history')

        # Add axis labels
        plt.ylabel('Metric')
        plt.xlabel('Epoch')

        # Add legend
        plt.legend(loc='upper left')

        # Save fig
        plt.savefig(save_path+'plot.png')

        # Show plot
        plt.show()
    except IOError:
        print "No X forwarding"
