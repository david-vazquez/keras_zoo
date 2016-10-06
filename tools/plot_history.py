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

    b_va_jaccard_0 = np.max(hist.history['0_val_jacc_percl'])
    b_va_jaccard2_0 = hist.history['0_val_jacc_percl'][best_index]
    b_va_jaccard_1 = np.max(hist.history['1_val_jacc_percl'])
    b_va_jaccard2_1 = hist.history['1_val_jacc_percl'][best_index]
    b_va_jaccard_2 = np.max(hist.history['2_val_jacc_percl'])
    b_va_jaccard2_2 = hist.history['2_val_jacc_percl'][best_index]
    b_va_jaccard_3 = np.max(hist.history['3_val_jacc_percl'])
    b_va_jaccard2_3 = hist.history['3_val_jacc_percl'][best_index]

    print ('Best epoch: {}. Train loss: {}. Val loss: {}. Val accuracy:{}. Val Jaccard: {}.'.format(best_index,
                                                                                                    b_tr_loss2,
                                                                                                    b_va_loss2,
                                                                                                    b_va_acc2,
                                                                                                    b_va_jaccard))
    print ('Val Jaccard: [{}, {}, {}, {}].'.format(b_va_jaccard2_0,
                                                   b_va_jaccard2_1,
                                                   b_va_jaccard2_2,
                                                   b_va_jaccard2_3))
    print ('Val Jaccard (best per class): [{}, {}, {}, {}].'.format(b_va_jaccard_0,
                                                                    b_va_jaccard_1,
                                                                    b_va_jaccard_2,
                                                                    b_va_jaccard_3))

    # Plot the metrics
    try:
        plot1(hist, save_path)
        plot2(hist, save_path)
    except IOError:
        print "No X forwarding"


# Plot training average metrics
def plot1(hist, save_path):
    # Compute the best values to add to the legend
    best_index = np.argmax(hist.history['val_jaccard'])
    b_tr_loss = np.min(hist.history['loss'])
    b_tr_loss2 = hist.history['loss'][best_index]
    b_va_loss = np.min(hist.history['val_loss'])
    b_va_loss2 = hist.history['val_loss'][best_index]
    b_va_jaccard = np.max(hist.history['val_jaccard'])
    b_va_acc = np.max(hist.history['val_acc'])
    b_va_acc2 = hist.history['val_acc'][best_index]

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
    plt.savefig(save_path+'plot1.png')

    # Show plot
    plt.show()


# Plot training metrics per class
def plot2(hist, save_path):
    best_index = np.argmax(hist.history['val_jaccard'])
    b_va_jaccard_0 = np.max(hist.history['0_val_jacc_percl'])
    b_va_jaccard2_0 = hist.history['0_val_jacc_percl'][best_index]
    b_va_jaccard_1 = np.max(hist.history['1_val_jacc_percl'])
    b_va_jaccard2_1 = hist.history['1_val_jacc_percl'][best_index]
    b_va_jaccard_2 = np.max(hist.history['2_val_jacc_percl'])
    b_va_jaccard2_2 = hist.history['2_val_jacc_percl'][best_index]
    b_va_jaccard_3 = np.max(hist.history['3_val_jacc_percl'])
    b_va_jaccard2_3 = hist.history['3_val_jacc_percl'][best_index]
    plt.plot(hist.history['0_val_jacc_percl'],
             label='Background Jaccard ({:.3f},{:.3f})'.format(b_va_jaccard_0,
                                                               b_va_jaccard2_0))
    plt.plot(hist.history['1_val_jacc_percl'],
             label='Polyp Jaccard ({:.3f},{:.3f})'.format(b_va_jaccard_1,
                                                          b_va_jaccard2_1))
    plt.plot(hist.history['2_val_jacc_percl'],
             label='Specularities Jaccard ({:.3f},{:.3f})'.format(b_va_jaccard_2,
                                                                  b_va_jaccard2_2))
    plt.plot(hist.history['3_val_jacc_percl'],
             label='Lumen Jaccard ({:.3f},{:.3f})'.format(b_va_jaccard_3,
                                                          b_va_jaccard2_3))

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
    plt.show()
