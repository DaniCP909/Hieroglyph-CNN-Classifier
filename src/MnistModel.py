
import cv2
import matplotlib.pyplot as plt

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from components.HieroglyphCharacterGenerator import HieroglyphCharacterGenerator
from components.HieroglyphAugmentator import HieroglyphAugmentator
from components.HieroglyphDataset import HieroglyphDataset

class MnistModel(nn.Module):
    def __init__(self, num_classes):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) #(input, output(n filters), kernel_size, stride)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 12 * 12, 1024) #input features from previous layer, reduce dim
        self.fc2 = nn.Linear(1024, num_classes) #output prev dense layer, n classes

    
    def forward(self, x):           #28x28                                  |   #64x64 57600                            |   #128x128 246016
        x = self.conv1(x)           #[bs, 1, 28, 28] --> [bs, 32, 26, 26]   |   #[bs, 1, 64, 64] --> [bs, 32, 62, 62]   |   #[bs, 1, 128, 128] --> [bs, 32, 126, 126]
        x = F.relu(x)
        x = self.conv2(x)           #[bs, 32, 26, 26] --> [bs, 64, 24, 24]  |   #[bs, 32, 62, 62] --> [bs, 64, 60, 60]  |   #[bs, 32, 126, 126] --> [bs, 64, 124, 124]
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      #[bs, 64, 24, 24] --> [bs, 64, 12, 12]  |   #[bs, 64, 60, 60] --> [bs, 64, 30, 30]  |   #[bs, 64, 124, 124] --> [bs, 64, 62, 62]
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
