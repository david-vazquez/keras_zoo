import os

from metrics.metrics import cce_flatt, IoU
from keras import backend as K
from keras.utils.visualize_util import plot

from models.fcn8 import build_fcn8
from models.segnet import build_segnet
from models.resnetFCN import build_resnetFCN
from models.densenetFCN import build_densenetFCN

from models.lenet import build_lenet
from models.alexNet import build_alexNet
from models.vgg import build_vgg
from models.resnet import build_resnet50
from models.inceptionV3 import build_inceptionV3


# Build the model
class Model_Factory():
    def __init__(self, cf, optimizer):
        self.cf = cf
        self.optimizer = optimizer


    def make(self):
        cf = self.cf
        optimizer = self.optimizer

        # Define the input size, loss and metrics
        if cf.dataset.class_mode == 'categorical':
            if K.image_dim_ordering() == 'th':
                in_shape = (cf.dataset.n_channels,
                            cf.target_size_train[0],
                            cf.target_size_train[1])
            else:
                in_shape = (cf.target_size_train[0],
                            cf.target_size_train[1],
                            cf.dataset.n_channels)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        elif cf.dataset.class_mode == 'segmentation':
            if K.image_dim_ordering() == 'th':
                in_shape = (cf.dataset.n_channels, None, None)
            else:
                in_shape = (None, None, cf.dataset.n_channels)
                #in_shape = (cf.target_size_train[0],
                #            cf.target_size_train[1],
                #            cf.dataset.n_channels)
            loss = cce_flatt(cf.dataset.void_class, cf.dataset.cb_weights)
            metrics = [IoU(cf.dataset.n_classes, cf.dataset.void_class)]
            # metrics = []
        else:
            raise ValueError('Unknown problem type')

        # Create the model
        if cf.model_name == 'fcn8':
            model = build_fcn8(in_shape, cf.dataset.n_classes, cf.weight_decay,
                               freeze_layers_from=cf.freeze_layers_from,
                               path_weights='weights/pascal-fcn8s-dag.mat')
        elif cf.model_name == 'segnet_basic':
            model = build_segnet(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                 freeze_layers_from=cf.freeze_layers_from,
                                 path_weights=None, basic=True)
        elif cf.model_name == 'segnet_vgg':
            model = build_segnet(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                 freeze_layers_from=cf.freeze_layers_from,
                                 path_weights=None, basic=False)
        elif cf.model_name == 'resnetFCN':
            model = build_resnetFCN(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                    freeze_layers_from=cf.freeze_layers_from,
                                    path_weights=None)
        elif cf.model_name == 'densenetFCN':
            model = build_densenetFCN(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                      freeze_layers_from=cf.freeze_layers_from,
                                      path_weights=None)
        elif cf.model_name == 'lenet':
            model = build_lenet(in_shape, cf.dataset.n_classes, cf.weight_decay)
        elif cf.model_name == 'alexNet':
            model = build_alexNet(in_shape, cf.dataset.n_classes, cf.weight_decay)
        elif cf.model_name == 'vgg16':
            model = build_vgg(in_shape, cf.dataset.n_classes, 16, cf.weight_decay,
                              load_pretrained=cf.load_imageNet,
                              freeze_layers_from=cf.freeze_layers_from)
        elif cf.model_name == 'vgg19':
            model = build_vgg(in_shape, cf.dataset.n_classes, 19, cf.weight_decay,
                              load_pretrained=cf.load_imageNet,
                              freeze_layers_from=cf.freeze_layers_from)
        elif cf.model_name == 'resnet50':
            model = build_resnet50(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                   load_pretrained=cf.load_imageNet,
                                   freeze_layers_from=cf.freeze_layers_from)
        elif cf.model_name == 'InceptionV3':
            model = build_inceptionV3(in_shape, cf.dataset.n_classes,
                                      cf.weight_decay,
                                      load_pretrained=cf.load_imageNet,
                                      freeze_layers_from=cf.freeze_layers_from)
        else:
            raise ValueError('Unknown model')

        # Load pretrained weights
        if cf.load_pretrained:
            print('   loading model weights from: ' + cf.weights_file + '...')
            model.load_weights(cf.weights_file, by_name=True)

        # Compile model
        model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

        # Show model structure
        if cf.show_model:
            model.summary()
            plot(model, to_file=os.path.join(cf.savepath, 'model.png'))

        # Output the model
        print ('   Model: ' + cf.model_name)
        return model
