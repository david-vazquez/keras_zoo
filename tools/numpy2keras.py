# Python import
import numpy as np

# Keras imports
from keras.models import Model


# Load numpy weights
def load_numpy(model, path_weights="weights/resnetFCN.npy"):
    print (' > Loading the numpy weights...')

    # Load weights
    weights_numpy = np.load(path_weights)
    weights_numpy = weights_numpy[()]

    # Show loaded layer names
    # for key, value in weights_numpy.items():
    #     print(key)

    # Iterate over model layers
    for layer in model.layers:
        # Get layer weights
        layer_weights = layer.get_weights()
        n_params = len(layer_weights)
        # print(' > name:{} - type:{} - n_params:{}'.format(layer.name, layer.__class__.__name__, n_params))

        # Check if this layer actualy has weights
        if n_params > 0:
            # Check if these weights are also in the numpy weights
            if (layer.name in weights_numpy):
                # Get this layer weights
                layer_weights_numpy = weights_numpy[layer.name]
                n_params_numpy = len(layer_weights_numpy)

                # Load and change scale and bias if it is a Batchnormalization
                if (layer.__class__.__name__ == 'BatchNormalization'):
                    scale_name = layer.name + '_scale'
                    layer_weights_numpy_scale = weights_numpy[scale_name]
                    layer_weights_numpy[0] = layer_weights_numpy_scale[0]
                    layer_weights_numpy[1] = layer_weights_numpy_scale[1]

                # Check that the model and numpy weights has the same number of params
                if (n_params_numpy != n_params):
                    raise ValueError('Number of parameters in layer {} is different for caffe ({}) and Keras({})'.format(layer.name, n_params_numpy, n_params))

                # Check that all the params have the same shape
                for i in range(n_params_numpy):
                    if (layer_weights_numpy[i].shape != layer_weights[i].shape):
                        print('Weight shape of parameter number {} in layer {} is different for caffe ({}) and Keras({})'.format(i, layer.name, layer_weights_numpy[i].shape, layer_weights[i].shape))

                # Set the weights
                layer.set_weights(layer_weights_numpy)

            else:
                print ('ERROR: ' + layer.name + ' not in caffe weights')

    return model
