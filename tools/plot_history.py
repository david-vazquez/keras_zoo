# Imports
import matplotlib.pyplot as plt
import numpy as np


# Plot training hisory
def plot_history(hist, save_path, n_classes):
    # Compute the best values to add to the legend
    best_index = np.argmax(hist['val_jaccard'])
    b_tr_loss = np.min(hist['loss'])
    b_tr_loss2 = hist['loss'][best_index]
    b_va_loss = np.min(hist['val_loss'])
    b_va_loss2 = hist['val_loss'][best_index]
    b_va_jaccard = np.max(hist['val_jaccard'])
    b_va_acc = np.max(hist['val_acc'])
    b_va_acc2 = hist['val_acc'][best_index]

    b_va_jaccard1 = []
    b_va_jaccard2 = []
    for i in range(n_classes):
        b_va_jaccard1 += [np.max(hist[str(i)+'_val_jacc_percl'])]
        b_va_jaccard2 += [hist[str(i)+'_val_jacc_percl'][best_index]]

    print ('Best epoch: {}. Train loss: {}. Val loss: {}. Val accuracy:{}. Val Jaccard: {}.'.format(best_index,
                                                                                                    b_tr_loss2,
                                                                                                    b_va_loss2,
                                                                                                    b_va_acc2,
                                                                                                    b_va_jaccard))
    print ('Val Jaccard: ' + str(b_va_jaccard2))
    print ('Val Jaccard(best per class): ' + str(b_va_jaccard1))

    # Plot the metrics
    try:
        plot1(hist, save_path)
        plot2(hist, save_path, n_classes)
    except IOError:
        print ("No X forwarding")


# Plot training average metrics
def plot1(hist, save_path):
    # Compute the best values to add to the legend
    best_index = np.argmax(hist['val_jaccard'])
    b_tr_loss = np.min(hist['loss'])
    b_tr_loss2 = hist['loss'][best_index]
    b_va_loss = np.min(hist['val_loss'])
    b_va_loss2 = hist['val_loss'][best_index]
    b_va_jaccard = np.max(hist['val_jaccard'])
    b_va_acc = np.max(hist['val_acc'])
    b_va_acc2 = hist['val_acc'][best_index]

    plt.plot(hist['loss'],
             label='train loss ({:.3f},{:.3f})'.format(b_tr_loss, b_tr_loss2))
    plt.plot(hist['val_loss'],
             label='valid loss ({:.3f},{:.3f})'.format(b_va_loss, b_va_loss2))
    plt.plot(hist['val_acc'],
             label='valid acc ({:.3f},{:.3f})'.format(b_va_acc, b_va_acc2))
    plt.plot(hist['val_jaccard'],
             label='valid jaccard ({:.3f})'.format(b_va_jaccard))

    # Add title
    plt.title('Model training history')

    # Add axis labels
    plt.ylabel('Metric')
    plt.xlabel('Epoch')

    # Add legend
    plt.legend(loc='upper left')

    # Save fig
    plt.savefig(save_path+'plot1.png')

    # Show plot
    # plt.show()
    plt.close()


# Plot training metrics per class
def plot2(hist, save_path, n_classes):
    best_index = np.argmax(hist['val_jaccard'])

    for i in range(n_classes):
        b_va_jaccard = np.max(hist[str(i)+'_val_jacc_percl'])
        b_va_jaccard2 = hist[str(i)+'_val_jacc_percl'][best_index]
        plt.plot(hist[str(i)+'_val_jacc_percl'],
                 label='Class{} Jaccard ({:.3f},{:.3f})'.format(i, b_va_jaccard,
                                                                   b_va_jaccard2))

    # Add title
    plt.title('Model training history (Jaccard per class)')

    # Add axis labels
    plt.ylabel('Metric')
    plt.xlabel('Epoch')

    # Add legend
    plt.legend(loc='upper left')

    # Save fig
    plt.savefig(save_path+'plot2.png')

    # Show plot
    # plt.show()
    plt.close()
