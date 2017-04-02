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
    
    # Colors (we assume there are no more than 7 metrics):
    colors = ['r', 'g', 'k', 'm', 'c', 'y', 'w']

    # Find the best epoch
    if best_type=='max':
        best_index = np.argmax(hist[best_metric])
    elif best_type=='min':
        best_index = np.argmin(hist[best_metric])
    else:
        raise ValueError('Unknown best type. It should be max or min')
    str += '   Best epoch: {}\n'.format(best_index)
    
    # Initialize figure:
    # Axis 1 will be for metrics, and axis 2 for losses.
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Check the training metrics
    idx = -1 # index for colors
    for metric in train_metrics:
        best_value = hist[metric][best_index]
        str += '   - {}: {}\n'.format(metric, best_value)
        if metric == 'loss':
            ax2.plot(hist[metric], 'b-', label='{} ({:.3f})'.format(metric, best_value))
        else:
            idx += 1
            ax1.plot(hist[metric], colors[idx] + '-', label='{} ({:.3f})'.format(metric, best_value))

    # Check the validation metrics
    idx = -1 # index for colors
    for metric in valid_metrics:
        best_value = hist[metric][best_index]
        str += '   - {}: {}\n'.format(metric, best_value)
        if metric == 'val_loss':
            ax2.plot(hist[metric], 'b--', label='{} ({:.3f})'.format(metric, best_value))
        else:
            idx += 1
            ax1.plot(hist[metric], colors[idx] + '--', label='{} ({:.3f})'.format(metric, best_value))

    # Print the result
    if verbose:
        print(str)
        
    ax1.set_ylim(0,1)

    # Add title
    plt.title('Model training history')

    # Add axis labels
    ax1.set_ylabel('Metric')
    ax2.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    
    # ??
    fig.tight_layout()

    # Add legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save fig
    plt.savefig(os.path.join(save_path, 'plot1.png'))

    # Close plot
    plt.close()
