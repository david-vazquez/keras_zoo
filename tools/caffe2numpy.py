import caffe
import numpy as np
import argparse

# Adapted from: https://github.com/qxcv/caffe2keras
# Inspired by: https://github.com/Lasagne/Recipes/blob/master/examples/Using%20a%20Caffe%20Pretrained%20Network%20-%20CIFAR10.ipynb


# Rotate weights (Correlation to convolution)
def rot90(W):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = np.rot90(W[i, j], 2)
    return W


# Converts the weights to numpy
def convert_weights(layers, v='V1'):
    weights = {}

    for name, layer in layers.items():
        typ = layer.type
        if typ == 'innerproduct':
            blobs = layer.blobs

            if (v == 'V1'):
                nb_filter = blobs[0].num
                stack_size = blobs[0].channels
                nb_col = blobs[0].height
                nb_row = blobs[0].width
            elif (v == 'V2'):
                if (len(blobs[0].shape.dim) == 4):
                    nb_filter = int(blobs[0].shape.dim[0])
                    stack_size = int(blobs[0].shape.dim[1])
                    nb_col = int(blobs[0].shape.dim[2])
                    nb_row = int(blobs[0].shape.dim[3])
                else:
                    nb_filter = 1
                    stack_size = 1
                    nb_col = int(blobs[0].shape.dim[0])
                    nb_row = int(blobs[0].shape.dim[1])
            else:
                raise RuntimeError('incorrect caffemodel version "' + v + '"')

            weights_p = np.array(blobs[0].data).reshape(
                nb_filter, stack_size, nb_col, nb_row)[0, 0, :, :]
            weights_p = weights_p.T  # need to swapaxes here, hence transpose. See comment in conv
            weights_b = np.array(blobs[1].data)
            layer_weights = [
                weights_p.astype(dtype=np.float32),
                weights_b.astype(dtype=np.float32)
            ]

            weights[layer.name] = layer_weights

        elif typ == 'BatchNorm':
            blobs = layer.blobs
            if (v == 'V2'):
                nb_kernels = int(blobs[0].shape.dim[0])
            else:
                nb_kernels = blobs[0].num

            weights_mean = np.array(blobs[0].data)
            weights_std_dev = np.array(blobs[1].data)

            weights[name] = [
                np.ones(nb_kernels), np.zeros(nb_kernels),
                weights_mean.astype(dtype=np.float32),
                weights_std_dev.astype(dtype=np.float32)
            ]
            print(" > {} {}: weights_mean={}, weights_std_dev={}".format(typ, name, weights_mean.shape, weights_std_dev.shape))

        elif typ == 'Scale':
            blobs = layer.blobs
            weights[name] = [
                np.array(blobs[0].data, dtype=np.float32), # Gamma (Scale factor)
                np.array(blobs[1].data, dtype=np.float32) # Beta (Bias term)
            ]
            print(" > {} {}: shape={} shape={}".format(typ, name, weights[name][0].shape, weights[name][1].shape))

        elif typ == 'Convolution':
            blobs = layer.blobs
            weights_p = rot90(np.array((blobs[0].data), dtype=np.float32))
            weights_p = np.transpose(weights_p, (2, 3, 1, 0))
            print(" > {} {}: weight shape={}".format(typ, name, weights_p.shape))
            if len(blobs) > 1:
                weights[name] = [weights_p, np.array(blobs[1].data, dtype=np.float32)]
            else:
                weights[name] = [weights_p]

        elif typ == 'Deconvolution':
            blobs = layer.blobs
            weights_p = rot90(np.array((blobs[0].data), dtype=np.float32))
            weights_p = np.transpose(weights_p, (2, 3, 1, 0))
            print(" > {} {}: weight shape={}".format(typ, name, weights_p.shape))
            if len(blobs) > 1:
                weights[name] = [weights_p, np.array(blobs[1].data, dtype=np.float32)]
            else:
                weights[name] = [weights_p]

        elif typ == 'ReLU':
            n_blobs = len(layer.blobs)
            if n_blobs > 0:
                print ('ERROR: ' + str(n_blobs))
            print(" > {} {}: No weights".format(typ, name))
        elif typ == 'Eltwise':
            n_blobs = len(layer.blobs)
            if n_blobs > 0:
                print ('ERROR: ' + str(n_blobs))
            print(" > {} {}: No weights".format(typ, name))
        elif typ == 'Split':
            n_blobs = len(layer.blobs)
            if n_blobs > 0:
                print ('ERROR: ' + str(n_blobs))
            print(" > {} {}: No weights".format(typ, name))
        elif typ == 'Dropout':
            n_blobs = len(layer.blobs)
            if n_blobs > 0:
                print ('ERROR: ' + str(n_blobs))
            print(" > {} {}: No weights".format(typ, name))
        elif typ == 'Crop':
            n_blobs = len(layer.blobs)
            if n_blobs > 0:
                print ('ERROR: ' + str(n_blobs))
            print(" > {} {}: No weights".format(typ, name))
        elif typ == 'Softmax':
            n_blobs = len(layer.blobs)
            if n_blobs > 0:
                print ('ERROR: ' + str(n_blobs))
            print(" > {} {}: No weights".format(typ, name))
        elif typ == 'Input':
            n_blobs = len(layer.blobs)
            if n_blobs > 0:
                print ('ERROR: ' + str(n_blobs))
            print(" > {} {}: No weights".format(typ, name))
        elif typ == 'Silence':
            n_blobs = len(layer.blobs)
            if n_blobs > 0:
                print ('ERROR: ' + str(n_blobs))
            print(" > {} {}: No weights".format(typ, name))
        else:
            print ('ERROR: Not found type: ' + typ)

    return weights


# Load caffe weights
def load_caffe(path_prototxt='weights/resnetFCN.prototxt',
               path_weights='weights/resnetFCN.caffemodel',
               out_path='weights/resnetFCN.npy',
               version='V1'):

    # Load the caffe network
    print (' --> Loading the caffe weights...')
    net_caffe = caffe.Net(path_prototxt, path_weights, caffe.TEST)
    layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers))

    # Convert weights
    print (' --> Converting the caffe weights to numpy...')
    weights_caffe = convert_weights(layers_caffe, v=version)

    # Save weights
    print (' --> Saving the weights in numpy...')
    np.save(out_path, weights_caffe)


# Entry point of the script
if __name__ == "__main__":
    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Caffe to numpy')
    parser.add_argument('-p', '--path_prototxt', type=str,
                        default='weights/resnetFCN.prototxt',
                        help='Path to the proto txt caffe file')
    parser.add_argument('-w', '--path_weights', type=str,
                        default='weights/resnetFCN.caffemodel',
                        help='Path to the caffe weights file')
    parser.add_argument('-o', '--out_path', type=str,
                        default='weights/resnetFCN.npy',
                        help='Output path')
    parser.add_argument('-v', '--version', type=int,
                        default=1, help='Caffe model version (1 or 2)')

    # Parse arguments
    arguments = parser.parse_args()
    path_prototxt = arguments.path_prototxt
    path_weights = arguments.path_weights
    out_path = arguments.out_path
    version = 'V' + str(arguments.version)

    # Call to the method
    load_caffe(path_prototxt, path_weights, out_path, version)
