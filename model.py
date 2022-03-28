import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # 110
        self.conv2 = nn.Conv2d(32, 64, 3)
        #54
        self.conv3= nn.Conv2d(64, 128, 3)
        #26
        self.conv4 = nn.Conv2d(128, 256, 3)
        #12


        self.bn1 = nn.BatchNorm2d(num_features=10)
        self.bn2 = nn.BatchNorm2d(num_features=20)
        self.bn3 = nn.BatchNorm2d(num_features=256)


        self.fc1 = nn.Linear(256*12*12, 1)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.bn3(x)


        # Prep for linear layer / Flatten
        x = x.view(x.size(0), -1)

        # linear layers with dropout in between
        x = F.sigmoid(self.fc1(x))



        return x