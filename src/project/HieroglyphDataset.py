import torch
from torch.utils.data import Dataset
from HieroglyphCharacterGenerator import HieroglyphCharacterGenerator
import cv2


class HieroglyphDataset(Dataset):

    #angle_sh: angulo o factor de sesgado(será *0.1)
    def __init__(self, generator, augmentator, mask=None, angle_sh=None, iterations=0):
        self.generator = generator
        self.augmentator = augmentator
        self.mask = mask
        self.angle_sh = angle_sh 
        
        if self.generator is None: 
            print("Error creating generator")

    
    def __len__(self):
        return self.generator.getFontLength() * self.augmentator.getAugmentatorLength()
    
    def __getitem__(self, idx):
        label = idx % self.generator.getFontLength()
        image = self.generator.getImageByLabel(label)
        image = self.augmentator.augment(image, idx, self.mask, self.angle_sh, 1)
        return (image, label)

