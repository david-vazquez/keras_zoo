from tools.data_loader import ImageDataGenerator


# Load datasets

class Dataset_Generators():
    def __init__(self):
        pass

    def make(self, cf):
        mean = cf.dataset.rgb_mean
        std = cf.dataset.rgb_std
        cf.dataset.cb_weights = None

        # Load training set
        print ('\n > Reading training set...')
        # Create the data generator with its data augmentation
        dg_tr = ImageDataGenerator(imageNet=cf.norm_imageNet_preprocess,
                                   rgb_mean=mean,
                                   rgb_std=std,
                                   rescale=cf.norm_rescale,
                                   featurewise_center=cf.norm_featurewise_center,
                                   featurewise_std_normalization=cf.norm_featurewise_std_normalization,
                                   samplewise_center=cf.norm_samplewise_center,
                                   samplewise_std_normalization=cf.norm_samplewise_std_normalization,
                                   gcn=cf.norm_gcn,
                                   zca_whitening=cf.norm_zca_whitening,
                                   crop_size=cf.crop_size_train,
                                   rotation_range=cf.da_rotation_range,
                                   width_shift_range=cf.da_width_shift_range,
                                   height_shift_range=cf.da_height_shift_range,
                                   shear_range=cf.da_shear_range,
                                   zoom_range=cf.da_zoom_range,
                                   channel_shift_range=cf.da_channel_shift_range,
                                   fill_mode=cf.da_fill_mode,
                                   cval=cf.da_cval,
                                   void_label=cf.dataset.void_class[0] if cf.dataset.void_class else None,
                                   horizontal_flip=cf.da_horizontal_flip,
                                   vertical_flip=cf.da_vertical_flip,
                                   spline_warp=cf.da_spline_warp,
                                   warp_sigma=cf.da_warp_sigma,
                                   warp_grid_size=cf.da_warp_grid_size,
                                   dim_ordering='th' if cf.model_name == 'yolo' else 'default',
                                   class_mode=cf.dataset.class_mode
                                   )

        # Compute normalization constants if required
        if cf.norm_fit_dataset:
            print ('   Computing normalization constants from training set...')
            # if cf.cb_weights_method is None:
            #     dg_tr.fit_from_directory(cf.dataset.path_train_img)
            # else:
            dg_tr.fit_from_directory(cf.dataset.path_train_img,
                                     cf.dataset.path_train_mask,
                                     len(cf.dataset.classes),
                                     cf.dataset.void_class,
                                     cf.cb_weights_method)

            mean = dg_tr.rgb_mean
            std = dg_tr.rgb_std
            cf.dataset.cb_weights = dg_tr.cb_weights

        # Load training data
        if not cf.dataset_name2:
            train_gen = dg_tr.flow_from_directory(directory=cf.dataset.path_train_img,
                                                  gt_directory=cf.dataset.path_train_mask,
                                                  resize=cf.resize_train,
                                                  target_size=cf.target_size_train,
                                                  color_mode=cf.dataset.color_mode,
                                                  classes=cf.dataset.classes,
                                                  class_mode=cf.dataset.class_mode,
                                                  batch_size=cf.batch_size_train,
                                                  shuffle=cf.shuffle_train,
                                                  seed=cf.seed_train,
                                                  save_to_dir=cf.savepath if cf.da_save_to_dir else None,
                                                  save_prefix='data_augmentation',
                                                  save_format='png')
        else:
            train_gen = dg_tr.flow_from_directory2(directory=cf.dataset.path_train_img,
                                                   gt_directory=cf.dataset.path_train_mask,
                                                   resize=cf.resize_train,
                                                   target_size=cf.target_size_train,
                                                   color_mode=cf.dataset.color_mode,
                                                   classes=cf.dataset.classes,
                                                   class_mode=cf.dataset.class_mode,
                                                   batch_size=int(cf.batch_size_train*(1.-cf.perc_mb2)),
                                                   shuffle=cf.shuffle_train,
                                                   seed=cf.seed_train,
                                                   save_to_dir=cf.savepath if cf.da_save_to_dir else None,
                                                   save_prefix='data_augmentation',
                                                   save_format='png',
                                                   directory2=cf.dataset2.path_train_img,
                                                   gt_directory2=cf.dataset2.path_train_mask,
                                                   batch_size2=int(cf.batch_size_train*cf.perc_mb2)
                                                   )

        # Load validation set
        print ('\n > Reading validation set...')
        dg_va = ImageDataGenerator(imageNet=cf.norm_imageNet_preprocess,
                                   rgb_mean=mean,
                                   rgb_std=std,
                                   rescale=cf.norm_rescale,
                                   featurewise_center=cf.norm_featurewise_center,
                                   featurewise_std_normalization=cf.norm_featurewise_std_normalization,
                                   samplewise_center=cf.norm_samplewise_center,
                                   samplewise_std_normalization=cf.norm_samplewise_std_normalization,
                                   gcn=cf.norm_gcn,
                                   zca_whitening=cf.norm_zca_whitening,
                                   crop_size=cf.crop_size_valid,
                                   dim_ordering='th' if cf.model_name == 'yolo' else 'default',
                                   class_mode=cf.dataset.class_mode)
        valid_gen = dg_va.flow_from_directory(directory=cf.dataset.path_valid_img,
                                              gt_directory=cf.dataset.path_valid_mask,
                                              resize=cf.resize_valid,
                                              target_size=cf.target_size_valid,
                                              color_mode=cf.dataset.color_mode,
                                              classes=cf.dataset.classes,
                                              class_mode=cf.dataset.class_mode,
                                              batch_size=cf.batch_size_valid,
                                              shuffle=cf.shuffle_valid,
                                              seed=cf.seed_valid)

        # Load testing set
        print ('\n > Reading testing set...')
        dg_ts = ImageDataGenerator(imageNet=cf.norm_imageNet_preprocess,
                                   rgb_mean=mean,
                                   rgb_std=std,
                                   rescale=cf.norm_rescale,
                                   featurewise_center=cf.norm_featurewise_center,
                                   featurewise_std_normalization=cf.norm_featurewise_std_normalization,
                                   samplewise_center=cf.norm_samplewise_center,
                                   samplewise_std_normalization=cf.norm_samplewise_std_normalization,
                                   gcn=cf.norm_gcn,
                                   zca_whitening=cf.norm_zca_whitening,
                                   dim_ordering='th' if cf.model_name == 'yolo' else 'default',
                                   class_mode=cf.dataset.class_mode)
        test_gen = dg_ts.flow_from_directory(directory=cf.dataset.path_test_img,
                                             gt_directory=cf.dataset.path_test_mask,
                                             resize=cf.resize_test,
                                             target_size=cf.target_size_test,
                                             color_mode=cf.dataset.color_mode,
                                             classes=cf.dataset.classes,
                                             class_mode=cf.dataset.class_mode,
                                             batch_size=cf.batch_size_test,
                                             shuffle=cf.shuffle_test,
                                             seed=cf.seed_test)

        return (train_gen, valid_gen, test_gen)
