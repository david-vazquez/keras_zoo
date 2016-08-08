# Deep learning for polyp characterization

## Results
| Model | Parameters | Train set | Test set | Tr. Cost | Val. Cost | Val. Acc | Val. Jacc | Test Acc | Test Jacc | Epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FCN8 | Rep+Conv, lr=1e-4, l2=1e-3 | CVC-612 | CVC-300 | ??.??  | ??.?? | ??.??%  | ??.??% | ??.??% | ??.??% | ?? |
| FCN8 | Rep+Conv, lr=1e-4, l2=1e-3 | CVC-612 | CVC-300 | ??.??  | ??.?? | ??.??%  | ??.??% | ??.??% | ??.??% | ?? |


## Posible problems to solve
 1. Polyp/No polyp image classification
 2. Polyp/Lumen/Specularity detection 
 3. Image semantic segmentation for the classes: Polyp/Lumen/Specularity/Void/Background
 4. Instance segmentation (Tracking of Polyps/Lumen/Specularities)
 5. Video Segmentacion
 6. Domain Adaptation
 7. Polyp classification (There are several types of polyps)
 
## Experiment configurations
 1. Intra pacient evaluation
 2. Inter pacient evaluation
 3. Inter camera evaluation
 4. Inter polyp evaluation
 
## First paper contributions
 1. Generate a public dataset that allows to perform different experiments
 2. Test a set of baselines for this dataset
  1, Jorge's published methods
  2. Deep learning methods: FCN8 y UNET
 
## Dataset
 1. Two datasets with: 300 and 612 color and fully annotated frames
 2. CVC-300 dataset is acquired at Irland and CVC-612 at Spain. The cameras are different.
 2. Image semantic annotation for each frame with the classes: Void(0)/Background(1)/Polyp(2)/Specularity(3)/Lumen(4)
 3. For each frame this extra metadata:
   1. Frame ID (In the dataset)
   2. Frame ID (In the original video) This data still is not available
   3. Patiend ID
   4. Polyp ID

## Framework
Keras

## Members
 1. Jorge Bernal
 2. Adriana Romero
 3. Michal Drozdzal
 4. David Vázquez
 5. People from Clinic? Jorge's Supervisor?
 7. Yoshua Bengio? Aaron Courville? Adriana's students?

## TODO
 - [X] Get Jorge's dataset with labels
 - [X] Adapt the dataset to work with FCN8 in lasagne
 - [X] First experiments with FCN8 in Lasagne
 - [ ] Experiments with FCN8 in Lasagne to get reasonable results
 - [ ] Init weights with Glorot for Relu or He
 - [ ] Weight the class contributions to be able of learning specularities (Very small)
 - [ ] Normalize images using mean and std??
 - [ ] Add data augmentation
 - [ ] Define a good split of the data for training/validation/test using the CSV files with the frames metadata
 - [ ] First experiments with Unet in Lasagne
 - [ ] Add DICE evaluations
 - [ ] Adapt the dataset to work in Keras
 - [ ] Upload FCN8 code for Keras
 - [ ] Move Unet model to Keras
 - [ ] Define experiments
 - [ ] Get Jorge's baselines for the proposed experiments
 - [ ] Select journal target (Michail has a proposal of a top medical imaging journal that answer in 1 month)
 - [ ] Add Frame ID (In the original video) to the datasets csv
 - [ ] Perform experiments
 - [ ] Write paper

## References jorge
 - [Impact of Image Preprocessing Methods on Polyp Localization in Colonoscopy Frames. Jorge Bernal et al.] (https://www.researchgate.net/profile/Jorge_Bernal5/publication/257602625_Impact_of_image_preprocessing_methods_on_polyp_localization_in_colonoscopy_frames/links/558924a208aed6bff80b3aa6.pdf) 
 - [WM-DOVA Maps for Accurate Polyp Highlighting in Colonoscopy: Validation vs. Saliency Maps from Physicians. Jorge Bernal et al.](http://158.109.8.37/files/BSF2015.pdf)
 - [Polyp Segmentation Method in Colonoscopy Videos by means of MSA-DOVA Energy Maps Calculation. Jorge Bernal et al.](http://158.109.8.37/files/BNS2014.pdf)

## References deep learning
 - [Fully Convolutional Networks for Semantic Segmentation. Jonathan Long et al.](https://arxiv.org/pdf/1411.4038.pdf)
 - [U-Net: Convolutional Networks for Biomedical Image Segmentation. Olaf Ronneberger et al.](https://arxiv.org/pdf/1505.04597.pdf)
