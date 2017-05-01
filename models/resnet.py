# Keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)

from keras.applications.resnet50 import ResNet50

# Paper: https://arxiv.org/abs/1512.03385

def build_resnet50(img_shape=(3, 224, 224), n_classes=1000, l2_reg=0.,
                load_pretrained=False, freeze_layers_from='base_model'):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None

    # Get base model
    base_model = ResNet50(include_top=False, weights=weights,
                       input_tensor=None, input_shape=img_shape)

    # Add final layers
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(n_classes, activation='softmax', name='fc1000')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('   Freezing base model layers')
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
               layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
               layer.trainable = True

    return model
