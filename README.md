# Keras implementation of Fully Convolutional Networks (FCN)

## Introduction

This repo contains the code to train and evaluate FCN8 network as described in [Fully Convolutional Neural Networks](https://arxiv.org/abs/1608.06993).

## Installation
You need to install :
- [Theano](https://github.com/Theano/Theano). Preferably the last version
- [Keras](https://github.com/fchollet/keras)
- (Recommend) [The new Theano GPU backend](https://github.com/Theano/libgpuarray). Compilation will be much faster.

## Run experiments
The architecture of the model is defined in fcn8.py. To train a model, you need to prepare the configuration in train file  where all the parameters needed for creating and training your model are precised.

To train a model, use the command : `THEANO_FLAGS='device=cuda0,floatX=float32' python train.py`. All the logs of the experiments are stored in the result folder of the experiment.

## Authors
David Vázquez, Jorge Bernal, F. Javier Sánchez, Gloria Fernández-Esparrach, Antonio M. López, Adriana Romero, Michal Drozdzal and Aaron Courville

## How to cite

## TODO
- [ ] Plot online
- [ ] Relaunch: Remember the number of the last epoch
- [ ] Find max batch size val-test
[ ] Split Synthia Train-Val
[ ] Experiment Synthia vs Synthia
[ ] Mix
[ ] Copy all synthia samples
[ ] Slurm
[ ] TensorFlow
[ ] Traffic sign Dataset
[ ] AlexNet sign Dataset
