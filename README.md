# Deep learning for polyp characterization

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
 2. Image sementic annotation for each frame with the classes: Polyp/Lumen/Specularity/Void/Background
 3. For each frame this extra metadata:
   1. Patiend ID
   2. Video ID
   3. Polyp ID
   4. Frame ID
   5. Camera ID
  
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
 - [ ] Adapt the dataset to work with FCN8 in lasagne
 - [ ] First experiments with FCN8 in Lasagne
 - [ ] First experiments with Unet in Lasagne
 - [ ] Add DICE evaluations
 - [ ] Adapt the dataset to work in Keras
 - [ ] Upload FCN8 code for Keras
 - [ ] Move Unet model to Keras
 - [ ] Define experiments
 - [ ] Get Jorge's baselines for the proposed experiments
 - [ ] Select journal target (Michail has a proposal of a top medical imaging journal that answer in 1 month)
 - [ ] Perform experiments
 - [ ] Write paper
