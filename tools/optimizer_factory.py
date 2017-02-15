from keras.optimizers import RMSprop

# Create the optimizer
class Optimizer_Factory():
    def __init__(self, cf):
        self.cf = cf
    
    def make(self):
        cf = self.cf
        
        # Create the optimizer
        if cf.optimizer == 'rmsprop':
            opt = RMSprop(lr=cf.learning_rate, rho=0.9, epsilon=1e-8, clipnorm=10)
            print ('   Optimizer: rmsprop. Lr: {}. Rho: 0.9, epsilon=1e-8, '
                   'clipnorm=10'.format(cf.learning_rate))
        else:
            raise ValueError('Unknown optimizer')

        # Return the optimizer
        return opt
