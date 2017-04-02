from keras.optimizers import (RMSprop, Adam, SGD)


# Create the optimizer
class Optimizer_Factory():
    def __init__(self):
        pass

    def make(self, cf):
        # Create the optimizer
        if cf.optimizer == 'rmsprop':
            opt = RMSprop(lr=cf.learning_rate, rho=0.9, epsilon=1e-8, clipnorm=10)
            print ('   Optimizer: rmsprop. Lr: {}. Rho: 0.9, epsilon=1e-8, '
                   'clipnorm=10'.format(cf.learning_rate))

        elif cf.optimizer == 'adam':
            opt = Adam(lr=cf.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        elif cf.optimizer == 'sgd':
            opt = SGD(lr=cf.learning_rate, momentum=0.9, nesterov=True)

        else:
            raise ValueError("Unknown optimizer. Valid optimizer arguments are: 'rmsprop', 'adam' and 'sgd'.")

        # Return the optimizer
        return opt
