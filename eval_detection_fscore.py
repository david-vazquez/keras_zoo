import os
import sys,time
import numpy as np
import math
import cv2

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

from models.yolo import build_yolo
from tools.yolo_utils import *

# Input parameters to select the Dataset and the model used
dataset_name = 'Udacity' #set to TT100K_detection otherwise
model_name = 'tiny-yolo' #set to yolo otherwise

# Net output post-processing needs two parameters:
detection_threshold = 0.6 # Min probablity for a prediction to be considered
nms_threshold       = 0.2 # Non Maximum Suppression threshold
# IMPORTANT: the values of these two params will affect the final performance of the netwrok
#            you are allowed to find their optimal values in the validation/train sets

if len(sys.argv) < 3:
  print "USAGE: python eval_detection_fscore.py weights_file path_to_images"
  quit()

if dataset_name == 'TT100K_detection':
    classes = ['i2','i4','i5','il100','il60','il80','io','ip','p10','p11','p12','p19','p23','p26','p27','p3','p5','p6','pg','ph4','ph4.5','ph5','pl100','pl120','pl20','pl30','pl40','pl5','pl50','pl60','pl70','pl80','pm20','pm30','pm55','pn','pne','po','pr40','w13','w32','w55','w57','w59','wo']
elif dataset_name == 'Udacity':
    classes = ['Car','Pedestrian','Truck']
else:
    print "Error: Dataset not found!"
    quit()

priors = [[0.9,1.2], [1.05,1.35], [2.15,2.55], [3.25,3.75], [5.35,5.1]]
input_shape = (3, 320, 320)

NUM_PRIORS  = len(priors)
NUM_CLASSES = len(classes)

if model_name == 'tiny-yolo':
    tiny_yolo = True
else:
    tiny_yolo = False

model = build_yolo(img_shape=input_shape,n_classes=NUM_CLASSES, n_priors=5,
               load_pretrained=False,freeze_layers_from='base_model',
               tiny=tiny_yolo)

model.load_weights(sys.argv[1])


test_dir = sys.argv[2]
imfiles = [os.path.join(test_dir,f) for f in os.listdir(test_dir) 
                                    if os.path.isfile(os.path.join(test_dir,f)) 
                                    and f.endswith('jpg')]

if len(imfiles) == 0:
  print "ERR: path_to_images do not contain any jpg file"
  quit()

inputs = []
img_paths = []
chunk_size = 128 # we are going to process all image files in chunks

ok = 0.
total_true = 0.
total_pred = 0.

for i,img_path in enumerate(imfiles):
  
  img = image.load_img(img_path, target_size=(input_shape[1], input_shape[2]))
  img = image.img_to_array(img)
  img = img / 255.
  inputs.append(img.copy())
  img_paths.append(img_path)

  if len(img_paths)%chunk_size == 0 or i+1 == len(imfiles):
    inputs = np.array(inputs)
    start_time = time.time()
    net_out = model.predict(inputs, batch_size=16, verbose=1)
    print ('{} images predicted in {:.5f} seconds. {:.5f} fps').format(len(inputs),time.time() - start_time,(len(inputs)/(time.time() - start_time)))

    # find correct detections (per image)
    for i,img_path in enumerate(img_paths):
        boxes_pred = yolo_postprocess_net_out(net_out[i], priors, classes, detection_threshold, nms_threshold)
        boxes_true = []
        label_path = img_path.replace('jpg','txt')
        gt = np.loadtxt(label_path)
        if len(gt.shape) == 1:
          gt = gt[np.newaxis,]
        for j in range(gt.shape[0]):
          bx = BoundBox(len(classes))
          bx.probs[int(gt[j,0])] = 1.
          bx.x, bx.y, bx.w, bx.h = gt[j,1:].tolist()
          boxes_true.append(bx)

        total_true += len(boxes_true)
        true_matched = np.zeros(len(boxes_true))
        for b in boxes_pred:
          if b.probs[np.argmax(b.probs)] < detection_threshold:
             continue
          total_pred += 1.
          for t,a in enumerate(boxes_true):
            if true_matched[t]:
              continue
            if box_iou(a, b) > 0.5 and np.argmax(a.probs) == np.argmax(b.probs):
              true_matched[t] = 1
              ok += 1.
              break
       
        # You can visualize/save per image results with this:
        #im = cv2.imread(img_path)
        #im = yolo_draw_detections(boxes_pred, im, priors, classes, detection_threshold, nms_threshold)
        #cv2.imshow('', im)
        #cv2.waitKey(0)


    inputs = []
    img_paths = []

    #print 'total_true:',total_true,' total_pred:',total_pred,' ok:',ok
    p = 0. if total_pred == 0 else (ok/total_pred)
    r = ok/total_true
    print('Precission = '+str(p))
    print('Recall     = '+str(r))
    f = 0. if (p+r) == 0 else (2*p*r/(p+r))
    print('F-score    = '+str(f))
