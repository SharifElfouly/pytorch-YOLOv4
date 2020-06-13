from PIL import Image
import time
import numpy as np
import cv2
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse

IMG_F = '/home/sharif/Downloads/public_place.jpg' 
CFG = '/home/sharif/Documents/pytorch-YOLOv4/cfg/yolov4.cfg' 
WEIGHTS = '/home/sharif/Documents/pytorch-YOLOv4/cfg/yolov4.weights' 

use_cuda = True
num_classes = 80
if num_classes == 20:
    namesfile = 'data/voc.names'
elif num_classes == 80:
    namesfile = 'data/coco.names'
else:
    namesfile = 'data/x.names'

def load_model():
    m = Darknet(CFG)
    m.load_weights(WEIGHTS)

    if use_cuda:
        m.cuda()
    return m

def detect(m, img):
    img = cv2.resize(img, (m.width, m.height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes = do_detect(m, img, 0.5, num_classes, 0.4, use_cuda)
    for box in boxes:
        x1,y1 = box[0],box[1]
        x2,y2 = box[2],box[3]
        c = box[-1]
        #print(c)
    return boxes

if __name__ == '__main__':
    m = load_model()
    img = Image.open(IMG_F)
    img = np.array(img)
    for i in range(1000):
        s = time.time()
        pred = detect(m, img)
        e = time.time()
        print(e-s)
