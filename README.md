# Keras implementation of Fully Convolutional Networks (FCN)

## Introduction

This repo contains the code to train and evaluate state of the art classification, detection and segmentation methods in a unified Keras framework working with Theano and TensorFlow. Pretrained models are also supplied. The available models are:

### Classification
 - [x] Lenet network as described in []().
 - [x] AlexNet network as described in []().
 - [x] VGG16 and VGG19 network as described in []().
 - [ ] GoogleNet network as described in []().
 - [ ] ResNet18 ResNet34 ResNet50 ResNet101 and ResNet152 network as described in []().
 
### Detection
 - [ ] YOLO network as described in []().
 - [ ] SSD network as described in []().
  
### Segmentation
 - [x] FCN8 network as described in [Fully Convolutional Neural Networks](https://arxiv.org/abs/1608.06993).
 - [ ] Segnet network as described in []().

It has wrappers for the followind datasets:
### Classification
 - [x] MIT dataset described in []().
 - [ ] ImageNet dataset described in []().
 - [ ] Pascal dataset described in []().
 
### Detection
 - [ ] TT100K dataset described in []().
  
### Segmentation
 - [ ] Camvid dataset described in []().
 - [x] Cityscapes dataset described in []().
 - [ ] Synthia dataset described in []().
 - [ ] Polyps dataset described in []().
 - [ ] Pascal dataset described in []().

## Installation
You need to install :
- [Theano](https://github.com/Theano/Theano) or [TensorFlow](https://github.com/Theano/Theano). Preferably the last version
- [Keras](https://github.com/fchollet/keras)

## Run experiments
The architecture of the model is defined in fcn8.py. To train a model, you need to prepare the configuration in train file  where all the parameters needed for creating and training your model are precised.

To train a model, use the command: `THEANO_FLAGS='device=cuda0,floatX=float32' python train.py`. All the logs of the experiments are stored in the result folder of the experiment.

## Authors
David Vázquez, Adriana Romero, Michal Drozdzal

## How to cite

## TODO
- [ ] Relaunch: Remember the number of the last epoch
- [ ] Mix
- [ ] Slurm
- [ ] TensorFlow
