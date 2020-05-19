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
        self.conv1 = nn.Conv2d(1,32,kernel_size=(4,4),padding=0)
        self.pool = nn.MaxPool2d(2,2,padding=0)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=0)
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(2,2),padding=0)
        self.dropout3 = nn.Dropout(0.3)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(1,1),padding=0)
        self.dropout4 = nn.Dropout(0.4)
        
        
        ## 2. It ends with a linear layer that represents the keypoints
#         self.fc1 = nn.Linear(10,6400)
        self.fc1 = nn.Linear(43264,1000) #works
        self.dropout5 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(6400,1000)
        self.fc2 = nn.Linear(1000,1000)
        self.dropout6 = nn.Dropout(0.6)
        self.fc3 = nn.Linear(1000,136)
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
#         print("the length after conv1",x.size())
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
#         print("length after conv2",x.size())
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)
#         print("length after conv3",x.size())
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout4(x)
#         print("length after conv4",x.size())
        
         #size of x is 256*13*13
#         print("length before flattening",x.size())
#         x = x.view(-1, x.size(0)) #flattening the layer
        x = x.view(x.size(0), -1)
#         print("THE SIZE OF X IS", x.size())
#         x = x.view(-1, self.num_flat_features(x))
#         print("length after flattening",x.size())
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        x = self.fc3(x)
        x = F.tanh(x) #changing softmax to tanh
#         print("length after softmax",x.size())
        # a modified x, having gone through all the layers of your model, should be returned
        return x
