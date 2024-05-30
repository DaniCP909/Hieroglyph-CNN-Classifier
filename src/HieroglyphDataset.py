import torch
from torch.utils.data import Dataset
from HieroglyphCharacterGenerator import HieroglyphCharacterGenerator
import cv2

class HieroglyphDataset(Dataset):
    def __init__(self, generator, augmentator, mask=None):
        self.generator = generator
        self.augmentator = augmentator
        
        if self.generator is None: 
            print("Error creating generator")

    
    def __len__(self):
        return (self.generator.getFontLength() + 1) * (self.augmentatos.getAugmentatorLength() + 1)
    
    def __getitem__(self, idx):
        label = (idx % (self.__len__() + 1))
        image = self.generator.getImageByLabel(label)
        image = self.augmentator.augment(image)
        return (image, label)

