import torch
from torch.utils.data import Dataset
from project.HieroglyphAugmentator import HieroglyphAugmentator
import cv2


class HieroglyphDataset(Dataset):

    #angle_sh: angulo o factor de sesgado(será *0.1)
    def __init__(self, generator_length: int, augmentator: HieroglyphAugmentator):
        self.generator_length = generator_length
        self.augmentator = augmentator
        

    
    def __len__(self):
        return (self.generator_length)
    
    def __getitem__(self, idx):
        label = idx
        image,_ = self.augmentator.augment(idx)
        
        return (image, label)

