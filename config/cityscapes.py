# Parameters
dataset_name      = 'cityscapes'    # Dataset name
model_name        = 'fcn8'          # FCN model to use
show_model        = False            # Show the architecture layers
plot_hist         = True            # Plot the training history after training
train_model       = True            # Train the model
test_model        = False           # Test the model

# Debug
debug             = False            # Use only few images for debuging
debug_images_train= 500             # N images for training in debug mode (-1 means all)
debug_images_valid= -1              # N images for validation in debug mode (-1 means all)
debug_images_test = 50              # N images for testing in debug mode (-1 means all)

# Noralization constants
input_norm        = 'rescale'       # Normalizations ['mean' | 'std' | 'meanAndStd' | 'rescale']
compute_constants = False           # If True it recompute std and mean from images. Either it uses the std and mean set at the dataset config file

# Class weight balance
cb_weights_method = None            # Label weight balance [None | 'median_freq_cost' | 'rare_freq_cost']

# Batch sizes
batch_size_train  = 10              # Batch size during training
batch_size_valid  = 30              # Batch size during validation
batch_size_test   = 30              # Batch size during testing
crop_size_train   = (224, 224)      # Crop size during training (Height, Width) or None
crop_size_valid   = None            # Crop size during validation
crop_size_test    = None            # Crop size during testing
resize_train      = (256, 512)      # Resize the image during training (Height, Width) or None
resize_valid      = (256, 512)      # Resize the image during validation
resize_test       = (256, 512)      # Resize the image during testing

# Data shuffle
shuffle_train     = True            # Whether to shuffle the training data
shuffle_valid     = False           # Whether to shuffle the validation data
shuffle_test      = False           # Whether to shuffle the testing data
seed_train        = 1924            # Random seed for the training shuffle
seed_valid        = 1924            # Random seed for the validation shuffle
seed_test         = 1924            # Random seed for the testing shuffle

# Training parameters
optimizer         = 'rmsprop'       # Optimizer
learning_rate     = 0.0001          # Training learning rate
weight_decay      = 0.              # Weight decay or L2 parameter norm penalty
n_epochs          = 1000            # Number of epochs during training
load_pretrained   = False           # Load a pretrained model for doing finetuning
weights_file      = 'weights.hdf5'  # Training weight file name

# Callback validation
valid_metrics                = ['val_loss', 'val_jaccard', 'val_acc', 'val_jaccard_perclass']

# Callback save results
save_results_enabled         = True            # Enable the Callback
save_results_nsamples        = 5               # Number of samples to save
save_results_batch_size      = 5               # Size of the batch

# Callback early stoping
earlyStopping_enabled        = True            # Enable the Callback
earlyStopping_monitor        = 'val_jaccard'   # Metric to monitor
earlyStopping_mode           = 'max'           # Mode ['max' | 'min']
earlyStopping_patience       = 100             # Max patience for the early stopping
earlyStopping_verbose        = 0               # Verbosity of the early stopping

# Callback model check point
checkpoint_enabled           = True            # Enable the Callback
checkpoint_monitor           = 'val_jaccard'   # Metric to monitor
checkpoint_mode              = 'max'           # Mode ['max' | 'min']
checkpoint_save_best_only    = True            # Save best or last model
checkpoint_save_weights_only = True            # Save only weights or also model
checkpoint_verbose           = 0              # Verbosity of the checkpoint

# Data augmentation for training
da_save_to_dir                   = False  # Save the images for debuging
da_featurewise_center            = False  # Substract mean - dataset
da_samplewise_center             = False  # Substract mean - sample
da_featurewise_std_normalization = False  # Divide std - dataset
da_samplewise_std_normalization  = False  # Divide std - sample
da_gcn                           = False  # Global contrast normalization
da_zca_whitening                 = False  # Apply ZCA whitening
da_rotation_range                = 0      # Rnd rotation degrees 0-180
da_width_shift_range             = 0.0    # Rnd horizontal shift
da_height_shift_range            = 0.0    # Rnd vertical shift
da_shear_range                   = 0.0    # Shear in radians
da_zoom_range                    = 0.0    # Zoom
da_channel_shift_range           = 0.     # Channecf.l shifts
da_fill_mode                     = 'constant'  # Fill mode
da_cval                          = 0.     # Void image value
da_horizontal_flip               = False  # Rnd horizontal flip
da_vertical_flip                 = False  # Rnd vertical flip
da_spline_warp                   = False  # Enable elastic deformation
da_warp_sigma                    = 10     # Elastic deformation sigma
da_warp_grid_size                = 3      # Elastic deformation gridSize
