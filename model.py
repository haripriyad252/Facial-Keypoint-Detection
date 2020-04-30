## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        self.conv1 = nn.conv2d(1,32,4)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.conv2d(32,64,3)
        self.conv3 = nn.conv2d(64,128,2)
        self.conv4 = nn.convd2(128,256,1)
        ## 2. It ends with a linear layer that represents the keypoints
        self.fc1 = nn.Linear(128,1000)
        self.fc2 = nn.Linear(1000,1000)
        self.fc3 = nn.Linear(1000,136)
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.pool(F.relu(self.conv1(x)))
        x = nn.Dropout2d(0.1)
        x = self.pool(F.relu(self.conv2(x)))
        x = nn.Dropout2d(0.2)
        x = self.pool(F.relu(self.conv3(x)))
        x = nn.Dropout2d(0.3)
        x = self.pool(F.relu(self.conv4(x)))
        x = nn.Dropout2d(0.4)
        
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = nn.Dropout(0.5)
        x = F.relu(self.fc2(x))
        x = nn.Dropout(0.6)
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
