"""
emnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

netdebug = False
EVT_SIZE = 11     # size of event grid (in X and Y)
ERR_SIZE = 60     # size of prediction grid (in X and Y)
PIXEL_SIZE = 0.005
PIXEL_ERR_RANGE_MIN = -0.0075  # error range minimum
PIXEL_ERR_RANGE_MAX = 0.0075   # error range maximum
chi = 128

class FCNet(nn.Module):

    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(EVT_SIZE*EVT_SIZE, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 2) #ERR_SIZE*ERR_SIZE)
        self.drop1 = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        if(netdebug): print("Init:",x.shape)
        x = torch.flatten(x, start_dim=1)

        if(netdebug): print("Flatten:",x.shape)
        x = self.fc1(x)

        if(netdebug): print("FC1:",x.shape)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)

        if(netdebug): print("FC2:",x.shape)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.fc3(x)

        if(netdebug): print("FC3:",x.shape)
        #x = self.sigmoid(x)

        return x

class basicCNN(nn.Module):
    def __init__(self):
        super(basicCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, chi, 4, padding=1)
        self.bn1   = nn.BatchNorm2d(chi)
        self.conv2 = nn.Conv2d(chi, chi*2, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(chi*2)
        self.conv3 = nn.Conv2d(chi*2, chi*4, 2, padding=1)
        self.bn3   = nn.BatchNorm2d(chi*4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(chi*4, ERR_SIZE*ERR_SIZE)
        self.drop1 = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if(netdebug): print(x.shape)
        x = self.pool4(self.bn1(F.relu(self.conv1(x))))
        if(netdebug): print(x.shape)
        x = self.pool3(self.bn2(F.relu(self.conv2(x))))
        if(netdebug): print(x.shape)
        x = self.pool2(self.bn3(F.relu(self.conv3(x))))
        if(netdebug): print(x.shape)
        x = x.flatten(start_dim=1)
        #x = x.view(-1, chi*16 * 1)
        if(netdebug): print(x.shape)
        x = self.drop1(x)
        x = self.fc(x)


        return x

# Basic CNN for regression-based solution.
class basicCNN_reg(nn.Module):
    def __init__(self):
        super(basicCNN_reg, self).__init__()

        self.conv1 = nn.Conv2d(1, chi, 4, padding=1)
        self.bn1   = nn.BatchNorm2d(chi)
        self.conv2 = nn.Conv2d(chi, chi*2, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(chi*2)
        self.conv3 = nn.Conv2d(chi*2, chi*4, 2, padding=1)
        self.bn3   = nn.BatchNorm2d(chi*4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc0 = nn.Linear(chi*4, 2)
        #self.fc1 = nn.Linear(ERR_SIZE*ERR_SIZE, 2)
        self.drop1 = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        if(netdebug): print(x.shape)
        x = self.pool4(self.bn1(F.leaky_relu(self.conv1(x))))
        if(netdebug): print(x.shape)
        x = self.pool3(self.bn2(F.leaky_relu(self.conv2(x))))
        if(netdebug): print(x.shape)
        x = self.pool2(self.bn3(F.leaky_relu(self.conv3(x))))
        if(netdebug): print(x.shape)
        x = x.flatten(start_dim=1)
        #x = x.view(-1, chi*16 * 1)
        if(netdebug): print(x.shape)
        x = self.drop1(x)
        # x = self.fc0(x)
        # x = self.drop1(x)
        x = self.fc0(x)
        #x = self.tanh(x)
        if(netdebug): print(x.shape)

        return x
