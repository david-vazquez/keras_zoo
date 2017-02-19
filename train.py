#!/usr/bin/env python
import argparse
import os
import sys
from getpass import getuser
import matplotlib
matplotlib.use('Agg')  # Faster plot

# Import tools
from config.configuration import Configuration
from tools.logger import Logger
from tools.dataset_generators import Dataset_Generators
from tools.optimizer_factory import Optimizer_Factory
from callbacks.callbacks_factory import Callbacks_Factory
from models.model_factory import Model_Factory


# Train the network
def process(cf):
    # Enable log file
    sys.stdout = Logger(cf.log_file)
    print (' ---> Init experiment: ' + cf.exp_name + ' <---')

    # Create the data generators
    train_gen, valid_gen, test_gen = Dataset_Generators().make(cf)

    # Create the optimizer
    print ('\n > Creating optimizer...')
    optimizer = Optimizer_Factory().make(cf)

    # Build model
    print ('\n > Building model...')
    model = Model_Factory().make(cf, optimizer)

    # Create the callbacks
    print ('\n > Creating callbacks...')
    cb = Callbacks_Factory().make(cf, valid_gen)

    if cf.train_model:
        # Train the model
        model.train(train_gen, valid_gen, cb)

    if cf.test_model:
        # Compute validation metrics
        model.test(valid_gen)
        # Compute test metrics
        model.test(test_gen)

    if cf.pred_model:
        # Compute validation metrics
        model.predict(valid_gen, tag='pred')
        # Compute test metrics
        model.predict(test_gen, tag='pred')

    # Finish
    print (' ---> Finish experiment: ' + cf.exp_name + ' <---')


# Sets the backend and GPU device.
class Environment():
    def __init__(self, backend='tensorflow'):
        #backend = 'tensorflow' # 'theano' or 'tensorflow'
        os.environ['KERAS_BACKEND'] = backend
        os.environ["CUDA_VISIBLE_DEVICES"]="0" # "" to run in CPU, extra slow! just for debuging
        if backend == 'theano':
            # os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu1,floatX=float32,optimizer=fast_compile'
            """ fast_compile que lo que hace es desactivar las optimizaciones => mas lento """
            os.environ['THEANO_FLAGS'] = 'device=gpu0,floatX=float32,lib.cnmem=0.95'
            print('Backend is Theano now')
        else:
            print('Backend is Tensorflow now')


# Main function
def main():
    # Define environment variables
    # Environment()

    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_path', type=str,
                        default=None, help='Configuration file')
    parser.add_argument('-e', '--exp_name', type=str,
                        default=None, help='Name of the experiment')
    parser.add_argument('-s', '--shared_path', type=str,
                        default='/data', help='Name of the experiment')
    parser.add_argument('-l', '--local_path', type=str,
                        default='/datatmp', help='Name of the experiment')

    arguments = parser.parse_args()

    assert arguments.config_path is not None, 'Please provide a configuration'\
                                              'path using -c config/pathname'\
                                              ' in the command line'
    assert arguments.exp_name is not None, 'Please provide a name for the '\
                                           'experiment using -e name in the '\
                                           'command line'

    # Define the user paths
    shared_path = arguments.shared_path
    local_path = arguments.local_path
    dataset_path = os.path.join(local_path, 'Datasets')
    shared_dataset_path = os.path.join(shared_path, 'Datasets')
    experiments_path = os.path.join(local_path, getuser(), 'Experiments')
    shared_experiments_path = os.path.join(shared_path, getuser(), 'Experiments')

    # Load configuration files
    configuration = Configuration(arguments.config_path, arguments.exp_name,
                                  dataset_path, shared_dataset_path,
                                  experiments_path, shared_experiments_path)
    cf = configuration.load()

    # Train /test/predict with the network, depending on the configuration
    process(cf)

    # Copy result to shared directory
    configuration.copy_to_shared()


# Entry point of the script
if __name__ == "__main__":
    main()
