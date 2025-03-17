import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import cv2

from HieroglyphCharacterGenerator import HieroglyphCharacterGenerator
from components.HieroglyphAugmentator import HieroglyphAugmentator
from HieroglyphDataset import HieroglyphDataset

cuda = torch.cuda.is_available()
print("GPU:", cuda)

paths = [ 
        "./files/fonts/Noto_Sans_Egyptian_Hieroglyphs/NotoSansEgyptianHieroglyphs-Regular.ttf",
        "./files/fonts/NewGardiner/NewGardinerBMP.ttf",
            ]
ranges = [ 
    (0x00013000, 0x0001342E),
    (0x0000E000, 0x0000E42E),
        ]

struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

generator = HieroglyphCharacterGenerator.HieroglyphCharacterGenerator(paths[0], ranges[0][0], ranges[0][1], font_size=100)
augmentator = HieroglyphAugmentator(generator)


dataset_hieroglyph = HieroglyphDataset(generator, augmentator=augmentator, mask=struct_element, angle_sh=2)
dataloader_hieroglyph = DataLoader(dataset_hieroglyph, batch_size=1000)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1,16,5) #(channels,output,kernel_size)   [Batch_size,1,28,28]  --> [Batch_size,16,24,24]
        self.mxp1 = nn.MaxPool2d(2)   #                                 [Batch_size,16,24,24] --> [Batch_size,16,24/2,24/2] --> [Batch_size,16,12,12]
        self.conv2 = nn.Conv2d(16,24,5) #                               [Batch_size,16,12,12] --> [Batch_size,24,8,8]
        self.mxp2 = nn.MaxPool2d(2)   #                                 [Batch_size,24,8,8] ---> [Batch_size,32,8/2,8/2] ---> [Batch_size,24,4,4]
        self.linear1 = nn.Linear(24 * 4 * 4, 100)                       #input shape --> 100 outputs
        self.linear2 = nn.Linear(100,1071)                                #100 inputs --> 10 outputs
        
    def forward(self,x):
        X = self.mxp1(F.relu(self.conv1(x)))
        X = self.mxp2(F.relu(self.conv2(X)))
        X = X.view(-1, 24 * 4 * 4)  #reshaping to input shape
        X = F.relu(self.linear1(X))
        X = self.linear2(X)
        return F.log_softmax(X, dim=1)

cnn = Model()

if cuda:
    cnn.cuda()