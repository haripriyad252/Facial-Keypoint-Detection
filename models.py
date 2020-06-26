import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        

        self.conv1 = nn.Conv2d(1,32,kernel_size=(4,4),padding=0)
        self.pool = nn.MaxPool2d(2,2,padding=0)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=0)
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(2,2),padding=0)
        self.dropout3 = nn.Dropout(0.3)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(1,1),padding=0)
        self.dropout4 = nn.Dropout(0.4)
        

        self.fc1 = nn.Linear(43264,1000) #works
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000,1000)
        self.dropout6 = nn.Dropout(0.6)
        self.fc3 = nn.Linear(1000,136)
                
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)

        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)

        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout4(x)


        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        x = self.fc3(x)
        x = F.tanh(x) #changing softmax to tanh
        # a modified x, having gone through all the layers of your model, should be returned
        return x
