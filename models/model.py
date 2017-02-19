import math
import time
import numpy as np
from keras.engine.training import GeneratorEnqueuer
#from model_factory import Model_Factory
from tools.save_images import save_img3


"""
Interface for normal (one net) models and adversarial models. Objects of
classes derived from Model are returned by method make() of the Model_Factory
class.
"""
class Model():
    def train(self, train_gen, valid_gen, cb):
        pass

    def predict(self, test_gen, tag='pred'):
        pass

    def test(self, test_gen):
        pass


"""
Wraper of regular models like FCN, SegNet etc consisting of a one Keras model.
But not GANs, which are made of two networks and have a different training
strategy.
In this class we implement the train(), test() and predict() methods common to
all of them.
"""
# TODO: Better call it Regular_Model ?
class One_Net_Model(Model):
    def __init__(self, model, cf, optimizer):
        self.cf = cf
        self.optimizer = optimizer
        self.model = model

    # Train the model
    def train(self, train_gen, valid_gen, cb):
        if (self.cf.train_model):
            print('\n > Training the model...')
            hist = self.model.fit_generator(generator=train_gen,
                                            samples_per_epoch=self.cf.dataset.n_images_train,
                                            nb_epoch=self.cf.n_epochs,
                                            verbose=1,
                                            callbacks=cb,
                                            validation_data=valid_gen,
                                            nb_val_samples=self.cf.dataset.n_images_valid,
                                            class_weight=None,
                                            max_q_size=10,
                                            nb_worker=1,
                                            pickle_safe=False)
            print('   Training finished.')

            return hist
        else:
            return None

    # Predict the model
    def predict(self, test_gen, tag='pred'):
        if self.cf.pred_model:
            print('\n > Predicting the model...')
            # Load best trained model
            # self.model.load_weights(os.path.join(self.cf.savepath, "weights.hdf5"))
            self.model.load_weights(self.cf.weights_file)

            # Create a data generator
            data_gen_queue, _stop, _generator_threads = GeneratorEnqueuer(self.test_gen, max_q_size=1)

            # Process the dataset
            start_time = time.time()
            for _ in range(int(math.ceil(self.cf.dataset.n_images_train/float(self.cf.batch_size_test)))):

                # Get data for this minibatch
                data = data_gen_queue.get()
                x_true = data[0]
                y_true = data[1].astype('int32')

                # Get prediction for this minibatch
                y_pred = self.model.predict(x_true)

                # Compute the argmax
                y_pred = np.argmax(y_pred, axis=1)

                # Reshape y_true
                y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[2],
                                             y_true.shape[3]))

                save_img3(x_true, y_true, y_pred, self.cf.savepath, 0,
                          self.cf.dataset.color_map, self.cf.dataset.classes, tag+str(_), self.cf.dataset.void_class)

            # Stop data generator
            _stop.set()

            total_time = time.time() - start_time
            fps = float(self.cf.dataset.n_images_test) / total_time
            s_p_f = total_time / float(self.cf.dataset.n_images_test)
            print ('   Predicting time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time, fps, s_p_f))

    # Test the model
    def test(self, test_gen):
        if self.cf.test_model:
            print('\n > Testing the model...')
            # Load best trained model
            self.model.load_weights(self.cf.weights_file)

            # Evaluate model
            start_time = time.time()
            test_metrics = self.model.evaluate_generator(test_gen,
                                                         self.cf.dataset.n_images_test,
                                                         max_q_size=10,
                                                         nb_worker=1,
                                                         pickle_safe=False)
            total_time = time.time() - start_time
            fps = float(self.cf.dataset.n_images_test) / total_time
            s_p_f = total_time / float(self.cf.dataset.n_images_test)
            print ('   Testing time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time, fps, s_p_f))

            # Compute Jaccard per class
            metrics_dict = dict(zip(self.model.metrics_names, test_metrics))
            I = np.zeros(self.cf.dataset.n_classes)
            U = np.zeros(self.cf.dataset.n_classes)
            jacc_percl = np.zeros(self.cf.dataset.n_classes)
            for i in range(self.cf.dataset.n_classes):
                I[i] = metrics_dict['I'+str(i)]
                U[i] = metrics_dict['U'+str(i)]
                jacc_percl[i] = I[i] / U[i]
                print ('   {:2d} ({:^15}): Jacc: {:6.2f}'.format(i,
                                                                 self.cf.dataset.classes[i],
                                                                 jacc_percl[i]*100))
            # Compute jaccard mean
            jacc_mean = np.nanmean(jacc_percl)
            print ('   Jaccard mean: {}'.format(jacc_mean))
