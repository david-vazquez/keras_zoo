# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:48:09 2017

@author: joans

Implementation of *the idea* in Semantic segmentation with adversarial networks,
P. Luc, C. Couprie S. Chintala, J. Verbeek. arXiv:1611:08408v1.

Segmentor network is segnet basic or segnet vgg. Discriminator is 5 encoding
blocks of VGG.
"""
import os
import numpy as np

# Keras imports
import keras.models as kmodels
from keras.layers import Input, Dense, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D, ZeroPadding2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation

from keras import backend as K
from keras.optimizers import RMSprop
from keras.utils.vis_utils import plot_model
from metrics.metrics import cce_flatt, IoU


from models.segnet import build_segnet
from model import Model

"""
This time the model is not a network but a pair of them, the segmentor (plays the
role of generator in GANs) and the discriminator. This means two Keras Model objects
instead of one. Training, testing, prediction are also different. So we make a
new class instead of a function build_whateverNet()
"""
class Adversarial_Semseg(Model):
    def __init__(self, cf, img_shape):
        # optimizer not passed at the moment, to complicated to adapt optimizers
        # factory because of 2 learning rates, 2 optimizers
        self.cf = cf
        self.img_shape = img_shape
        self.n_classes = cf.dataset.n_classes

        # make and compile the two models
        self.segmentor = self.make_segmentor()
        self.discriminator = self.make_discriminator()

        # Show model structure
        if self.cf.show_model:
            print('Segmentor')
            self.segmentor.summary()
            plot(self.segmentor, to_file=os.path.join(cf.savepath, 'model_segmentor.png'))
            print('Discriminator')
            self.discriminator.summary()
            plot(self.discriminator, to_file=os.path.join(cf.savepath, 'model_discriminator.png'))


    def make_segmentor(self):
        segmentor = build_segnet(self.img_shape, self.n_classes,
                                 l2_reg=0., init='glorot_uniform', path_weights=None,
                                 freeze_layers_from=None, use_unpool=False, basic=False)
        lr = 1e-04
        optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1e-8, clipnorm=10)
        print ('   Optimizer segmentor: rmsprop. Lr: {}. Rho: 0.9, epsilon=1e-8, '
               'clipnorm=10'.format(lr))
        the_loss = cce_flatt(self.cf.dataset.void_class, self.cf.dataset.cb_weights)
        metrics = [IoU(self.cf.dataset.n_classes, self.cf.dataset.void_class)]

        segmentor.compile(loss=the_loss, metrics=metrics, optimizer=optimizer)
        return segmentor


    def make_discriminator(self):
        # TODO just to have something, 5 layers vgg-like
        inputs = Input(shape=self.img_shape)
        enc1 = self.downsampling_block_basic(inputs, 64, 7)
        enc2 = self.downsampling_block_basic(enc1,   64, 7)
        enc3 = self.downsampling_block_basic(enc2,   92, 7)
        enc4 = self.downsampling_block_basic(enc3,  128, 7)
        enc5 = self.downsampling_block_basic(enc4,  128, 7)
        flat = Flatten()(enc5)
        dense1 = Dense(512, activation='sigmoid')(flat)
        dense2 = Dense(512, activation='sigmoid')(dense1)
        fake = Dense(1, activation='sigmoid', name='generation')(dense2)
        # Dense(2,... two classes : real and fake
        # change last activation to softmax ?
        discriminator = kmodels.Model(input=inputs, output=fake)

        lr = 1e-04
        optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1e-8, clipnorm=10)
        print ('   Optimizer discriminator: rmsprop. Lr: {}. Rho: 0.9, epsilon=1e-8, '
               'clipnorm=10'.format(lr))

        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        # TODO metrics=metrics,
        return discriminator



    def channel_idx(self):
        dim_ordering = K.image_dim_ordering()
        if dim_ordering == 'th':
            return 1
        else:
            return 3


    # To build the discriminator, its a block from SegNet
    def downsampling_block_basic(self, inputs, n_filters, filter_size,
                                 W_regularizer=None):
        pad = ZeroPadding2D(padding=(1, 1))(inputs)
        conv = Convolution2D(n_filters, filter_size, filter_size,
                             border_mode='same', W_regularizer=W_regularizer)(pad)
        bn = BatchNormalization(mode=0, axis=self.channel_idx())(conv)
        act = Activation('relu')(bn)
        maxp = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act)
        return maxp



    def train(self, train_gen, valid_gen, cb):
        pass
        # TODO

    def predict(self, test_gen, tag='pred'):
        pass
        # TODO

    def test(self, test_gen):
        pass
        # TODO



if __name__ == '__main__':
    img_shape = (360, 480, 3)
    class cf:
        show_model = True
        savepath = './'
        class dataset:
            n_classes = 12
            void_class = [11]
            cb_weights = None

    print ('BUILD AND COMPILING')
    adv = Adversarial_Semseg(cf, img_shape)
    print ('END OF BUILDING AND COMPILING')
    adv.segmentor.summary()
    adv.discriminator.summary()
