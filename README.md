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
  
