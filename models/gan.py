import os
import numpy as np
import random
# Keras imports
import keras.models as kmodels
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.utils.visualize_util import plot
from model import Model
from generator import build_generator
from discriminator import build_discriminator
from tqdm import tqdm
import matplotlib.pyplot as plt


class GAN(Model):
    def __init__(self, cf, img_shape):
        self.cf = cf
        self.img_shape = img_shape

        # Make and compile the generator
        self.g_lr = 1e-04
        self.g_optimizer = Adam(lr=self.g_lr)
        self.g_img_shape = [100]
        self.generator = self.make_generator(self.g_img_shape,
                                             self.g_optimizer,
                                             the_loss='binary_crossentropy',
                                             metrics=[])

        # Make and compile the discriminator
        self.d_lr = 1e-03
        self.d_optimizer = Adam(lr=self.d_lr)
        self.d_img_shape = img_shape
        self.discriminator = self.make_discriminator(self.d_img_shape,
                                                     self.d_optimizer,
                                                     the_loss='categorical_crossentropy',
                                                     metrics=[])

        # Freeze weights in the discriminator for stacked training
        self.make_trainable(self.discriminator, False)

        # Make and compile the GAN
        self.GAN = self.make_gan(self.g_img_shape, self.d_optimizer,
                                 the_loss='categorical_crossentropy',
                                 metrics=[])

    # Make generator
    def make_generator(self, img_shape, optimizer,
                       the_loss='binary_crossentropy', metrics=[]):
        # Build model
        generator = build_generator(img_shape, n_channels=200, l2_reg=0.)

        # Compile model
        generator.compile(loss=the_loss, metrics=metrics, optimizer=optimizer)

        # Show model
        if self.cf.show_model:
            print('Generator')
            generator.summary()
            plot(generator,
                 to_file=os.path.join(self.cf.savepath, 'model_generator.png'))
        return generator

    # Make discriminator
    def make_discriminator(self, img_shape, optimizer,
                           the_loss='categorical_crossentropy', metrics=[]):
        # Build model
        discriminator = build_discriminator(img_shape, dropout_rate=0.25,
                                            l2_reg=0.)

        # Compile model
        discriminator.compile(loss=the_loss, metrics=metrics,
                              optimizer=optimizer)

        # Show model
        if self.cf.show_model:
            print('Discriminator')
            discriminator.summary()
            plot(discriminator,
                 to_file=os.path.join(self.cf.savepath,
                                      'model_discriminator.png'))

        return discriminator

    # Make GAN
    def make_gan(self, img_shape, optimizer,
                 the_loss='categorical_crossentropy', metrics=[]):
        # Build stacked GAN model
        gan_input = Input(shape=img_shape)
        H = self.generator(gan_input)
        gan_V = self.discriminator(H)
        GAN = kmodels.Model(gan_input, gan_V)

        # Compile model
        GAN.compile(loss=the_loss, metrics=metrics, optimizer=optimizer)

        # Show model
        if self.cf.show_model:
            print('GAN')
            GAN.summary()
            plot(GAN, to_file=os.path.join(self.cf.savepath, 'model_GAN.png'))

        return GAN

    # Make the network trainable or not
    def make_trainable(self, net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val

    def channel_idx(self):
        dim_ordering = K.image_dim_ordering()
        if dim_ordering == 'th':
            return 1
        else:
            return 3

    def pretrain_discriminator(self, x_train, n_samples=10000, batch_size=128,
                               nb_epoch=1):
        # Get n random samples from the real training images
        train_idx = random.sample(range(0, x_train.shape[0]), n_samples)
        real_images = x_train[train_idx, :, :, :]
        # print (real_images.shape)

        # Generate n random fake samples
        noise_gen = np.random.uniform(0, 1, size=[n_samples, self.g_img_shape[0]])
        generated_images = self.generator.predict(noise_gen)
        # print (generated_images.shape)

        # Concatenate the real and generated images into a training set
        x = np.concatenate((real_images, generated_images))

        # Create the one-hot encoding of the labels
        y = np.zeros([2*n_samples, 2])
        y[:n_samples, 1] = 1
        y[n_samples:, 0] = 1

        # Pre-train the discriminator network
        self.make_trainable(self.discriminator, True)
        self.discriminator.fit(x, y, nb_epoch=nb_epoch, batch_size=batch_size)
        y_hat = self.discriminator.predict(x)

        # print (y_hat.shape)

        y_hat_idx = np.argmax(y_hat, axis=1)
        y_idx = np.argmax(y, axis=1)
        diff = y_idx-y_hat_idx
        n_tot = y.shape[0]
        n_rig = (diff == 0).sum()
        acc = n_rig*100.0/n_tot
        print "Accuracy: %0.02f pct (%d of %d) right" % (acc, n_rig, n_tot)

    def plot_loss(self, losses):
        # display.clear_output(wait=True)
        # display.display(plt.gcf())
        plt.figure(figsize=(10, 8))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.legend()
        # plt.show()
        # Save fig
        plt.savefig(os.path.join(self.cf.savepath, 'plot_loss.png'))

        # Close plot
        plt.close()

    def plot_gen(self, n_samples=16, dim=(4, 4), figsize=(10, 10)):
        noise = np.random.uniform(0, 1, size=[n_samples, self.g_img_shape[0]])
        generated_images = self.generator.predict(noise)
        # print (generated_images.shape)
        generated_images = np.squeeze(generated_images, axis=self.channel_idx())
        # print (generated_images.shape)

        plt.figure(figsize=figsize)
        for i in range(n_samples):
            plt.subplot(dim[0], dim[1], i+1)
            img = generated_images[i, :, :]
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        # plt.show()
        # Save fig
        plt.savefig(os.path.join(self.cf.savepath, 'plot_gen.png'))

        # Close plot
        plt.close()

    # Train GAN
    def train_for_n(self, x_train, nb_epoch=5000, plt_frq=25, batch_size=32):

        # set up loss storage vector
        losses = {"d": [], "g": []}

        for e in tqdm(range(nb_epoch)):

            # Get n random samples from the real training images
            train_idx = np.random.randint(0, x_train.shape[0], size=batch_size)
            real_images = x_train[train_idx, :, :, :]
            # print (real_images.shape)

            # Generate n random fake samples
            noise_gen = np.random.uniform(0, 1, size=[batch_size, self.g_img_shape[0]])
            generated_images = self.generator.predict(noise_gen)
            # print (generated_images.shape)

            # Concatenate the real and generated images into a training set
            x = np.concatenate((real_images, generated_images))

            # Create the one-hot encoding of the labels
            y = np.zeros([2*batch_size, 2])
            y[:batch_size, 1] = 1  # Real
            y[batch_size:, 0] = 1  # Generated

            # Train the discriminator
            # self.make_trainable(self.discriminator, True)
            d_loss = self.discriminator.train_on_batch(x, y)
            losses["d"].append(d_loss)

            # train Generator-Discriminator stack on input noise to non-generated output class
            noise_tr = np.random.uniform(0, 1, size=[batch_size, self.g_img_shape[0]])
            y2 = np.zeros([batch_size, 2])
            y2[:, 1] = 1 # Real

            # self.make_trainable(self.discriminator, False)
            g_loss = self.GAN.train_on_batch(noise_tr, y2)
            losses["g"].append(g_loss)

            # Updates plots
            if e % plt_frq == plt_frq-1:
                self.plot_loss(losses)
                self.plot_gen()

    def load_data(self, load_data_func):
        # Load data
        (x_train, y_train), (x_test, y_test) = load_data_func(os.path.join(self.cf.dataset.path, self.cf.dataset.dataset_name+'.pkl.gz'))
        img_rows, img_cols = self.cf.dataset.img_shape

        # Reshape
        if K.image_dim_ordering() == 'th':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        elif K.image_dim_ordering() == 'tf':
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        # Normalize
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # One hot encoding
        y_train2 = np.zeros((len(y_train), self.cf.dataset.n_classes), dtype='float32')
        for i, label in enumerate(y_train):
            y_train2[i, label] = 1.
        y_train = y_train2

        return (x_train, y_train), (x_test, y_test)

    def train2(self, load_data_func, cb):
        if (self.cf.train_model):
            print('\n > Training the model...')
            # Load data
            (x_train, y_train), (x_test, y_test) = self.load_data(load_data_func)

            # Pretrain discriminator
            self.pretrain_discriminator(x_train, n_samples=10000,
                                        batch_size=128, nb_epoch=1)

            self.train_for_n(x_train, nb_epoch=5000, plt_frq=25, batch_size=32)

            print('   Training finished.')

            # return hist
            return None
        else:
            return None

    def train(self, train_gen, valid_gen, cb):
        pass
        # TODO

    def predict(self, test_gen, tag='pred'):
        pass
        # TODO

    def test(self, test_gen):
        pass
        # TODO
