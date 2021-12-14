import torch
import torch.nn as nn
import numpy as np

from darknet import *

class YoloV3Net (nn.Module): 
    def __init__ (self, num_anchors = 3, num_classes = 80): 
        super (YoloV3Net, self).__init__ ()
        self.anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]] 
        self.num_anchors = num_anchors
        self.num_classes = num_classes      
        self.darknet = Darknet_53 ()
        self.cnn = nn.Sequential (OrderedDict ([
               ('conv0', nn.Sequential ( 
                         Conv2D (1024, 512, 1, 1), 
                         Conv2D (512, 1024, 3, 1), 
                         Conv2D (1024, 512, 1, 1),
                         Conv2D (512, 1024, 3, 1), 
                         Conv2D (1024, 512, 1, 1))),

               ('conv1', nn.Sequential(
                         Conv2D (512, 1024, 3, 1), 
                         nn.Conv2d (1024, 255, 1, 1))), 

               ('conv2', nn.Sequential ( 
                         Conv2D (512, 256, 1, 1), 
                         nn.UpsamplingNearest2d (scale_factor = 2))), 

               ('conv3', nn.Sequential ( 
                         Conv2D (768, 256, 1, 1), 
                         Conv2D (256, 512, 3, 1), 
                         Conv2D (512, 256, 1, 1), 
                         Conv2D (256, 512, 3, 1), 
                         Conv2D (512, 256, 1, 1))), 

               ('conv4', nn.Sequential ( 
                         Conv2D (256, 512, 3, 1), 
                         nn.Conv2d (512, 255, 1, 1))), 

               ('conv5', nn.Sequential ( 
                         Conv2D (256, 128, 1, 1), 
                         nn.UpsamplingNearest2d (scale_factor = 2))), 

               ('conv6', nn.Sequential ( 
                         Conv2D (384, 128, 1, 1), 
                         Conv2D (128, 256, 3, 1), 
                         Conv2D (256, 128, 1, 1), 
                         Conv2D (128, 256, 3, 1), 
                         Conv2D (256, 128, 1, 1))),

               ('conv7', nn.Sequential (
                         Conv2D (128, 256, 3, 1), 
                         nn.Conv2d (256, 255, 1, 1)))]))

    def forward (self, x):
        route36, route61, x = self.darknet (x)
        route79 = self.cnn [0] (x)
        yolo82  = self.cnn [1] (route79)
        x       = self.cnn [2] (route79)
        x       = torch.cat ([x, route61], 1)
        route91 = self.cnn [3] (x)
        yolo94  = self.cnn [4] (route91)
        x       = self.cnn [5] (route91)
        x       = torch.cat ([x, route36], 1)
        x       = self.cnn [6] (x)
        yolo106 = self.cnn [7] (x)
        return yolo82, yolo94, yolo106


def load_weights (model, wt_file):
    """ load weights from .weights file """
    buf = np.fromfile (wt_file, dtype = np.float32)
    start = 5
 
    #darknet
    start = load_conv_bn (buf,start, model.darknet.cnn [0].conv, model.darknet.cnn [0].bn)
    n = [2, 3, 9, 9, 5]
    for i in range (1, 6):
        start = load_conv_bn (buf, start, model.darknet.cnn [i] [0].conv, model.darknet.cnn [i] [0].bn)
        for j in range (1, n [i - 1]):
            for k in range (2):
                start = load_conv_bn (buf, start, model.darknet.cnn [i] [j].conv [k].conv, model.darknet.cnn [i] [j].conv [k].bn)
 
    m = [5, 1, 1, 5, 1, 1, 5, 1]
    for i in range (8):
        for j in range (m [i]):
            if i  in [1, 4, 7]:
                start = load_conv_bn (buf, start, model.cnn [i] [0].conv, model.cnn [i] [0].bn)
                start = load_conv (buf, start, model.cnn [i] [1])
            else:
                start = load_conv_bn (buf, start, model.cnn [i] [j].conv, model.cnn [i] [j].bn)
 
    return start

use_gpu = False

def load_model (weights, device):
    """ load model and it's weights """
    model = YoloV3Net ()
   
    load_weights (model, weights)
    #model.load_state_dict (torch.load (weights))
    return model.to(device)