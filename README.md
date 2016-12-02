# A Benchmark for Endoluminal Scene Segmentation of Colonoscopy Images

## Introduction

This repo contains the code to train and evaluate FCN8 network as described in [A Benchmark for Endoluminal Scene Segmentation of Colonoscopy Images](https://arxiv.org/submit/1741331). We investigate the use of [Fully Convolutional Neural Networks](https://arxiv.org/abs/1608.06993) for Endoluminal Scene Segmentation, and report state of the art results on EndoScene dataset.

## Installation

You need to install :
- [Theano](https://github.com/Theano/Theano). Preferably the last version
- [Keras](https://github.com/fchollet/keras)
- The dataset(http://adas.cvc.uab.es/endoscene)
- (Recommend) [The new Theano GPU backend](https://github.com/Theano/libgpuarray). Compilation will be much faster.

## Run experiments

The architecture of the model is defined in fcn8.py. To train a model, you need to prepare the configuration in train file  where all the parameters needed for creating and training your model are precised.

To train a model, use the command : `THEANO_FLAGS='device=cuda0,floatX=float32' python train.py`. All the logs of the experiments are stored in the result folder of the experiment.

## Authors
David Vázquez, Jorge Bernal, F. Javier Sánchez, Gloria Fernández-Esparrach, Antonio M. López, Adriana Romero, Michal Drozdzal and Aaron Courville

## How to cite
