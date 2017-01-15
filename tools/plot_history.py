# Imports
import matplotlib.pyplot as plt
import numpy as np
import os
import time


# Plot training hisory
def plot_history(hist, save_path, n_classes,
                 train_metrics=['loss', 'acc'],
                 valid_metrics=['val_loss', 'val_acc'],
                 best_metric='val_jaccard',
                 best_type='max',
                 verbose=True):

    # Create string to print
    str = ''

    # Find the best epoch
    if best_type=='max':
        best_index = np.argmax(hist[best_metric])
    elif best_type=='min':
        best_index = np.argmin(hist[best_metric])
    else:
        raise ValueError('Unknown best type. It should be max or min')
    str += '   Best epoch: {}\n'.format(best_index)

    # Check the training metrics
    for metric in train_metrics:
        best_value = hist[metric][best_index]
        str += '   - {}: {}\n'.format(metric, best_value)
        plt.plot(hist[metric], label='{} ({:.3f})'.format(metric, best_value))

    # Check the training metrics
    for metric in valid_metrics:
        best_value = hist[metric][best_index]
        str += '   - {}: {}\n'.format(metric, best_value)
        plt.plot(hist[metric], label='{} ({:.3f})'.format(metric, best_value))

    # Print the result
    if verbose:
        print(str)

    # Add title
    plt.title('Model training history')

    # Add axis labels
    plt.ylabel('Metric')
    plt.xlabel('Epoch')

    # Add legend
    plt.legend(loc='upper left')

    # Save fig
    plt.savefig(os.path.join(save_path, 'plot1.png'))

    # Close plot
    plt.close()
