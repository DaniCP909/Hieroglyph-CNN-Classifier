
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

class ModMnistModel(nn.Module):
    def __init__(self, num_classes, fc1_out=2048):
        super(ModMnistModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, 1) #(input, output(n filters), kernel_size, stride)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1) #512

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256 * 22 * 22, fc1_out) 
        self.fc2 = nn.Linear(fc1_out, num_classes) 

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)
        #
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
