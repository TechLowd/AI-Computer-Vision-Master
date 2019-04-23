## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.conv1 = nn.Conv2d(1, 32, 5) 
        # Pooling
        self.avgPool = nn.AvgPool2d(2, 2)    
        self.mxPool = nn.MaxPool2d(2, 2)
        # Conv.
        self.conv2 = nn.Conv2d(32, 64, 3)     
        self.conv3 = nn.Conv2d(64, 128, 3)   
        self.conv4 = nn.Conv2d(128, 256, 3)   
        # Linear
        self.fc = nn.Linear(256*12*12, 68*2)
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x))): 224
        # 2 Conv. & Pooling Layers with Relu
        # AvgPool: to blur, reduce Noise; mxPool: to sharpen
        x = self.avgPool(F.relu(self.conv1(x)))     # (32, 220, 220), (32, 110, 110)
        x = self.mxPool(F.relu(self.conv2(x)))      # (64, 108, 108), (64, 54, 54)
        x = self.mxPool(F.relu(self.conv3(x)))      # (128, 52, 52), (128, 26, 26)
        x = self.mxPool(F.relu(self.conv4(x)))      # (256, 24, 24), (256, 12, 12)
        # Flaten
        x = x.view(x.size(0), -1) 
        x = self.fc(x)                              # No Act. scale [-1, 1]
        # a modified x, having gone through all the layers of your model, should be returned
        return x
