import torch
from torch.utils.data import Dataset
from HieroglyphAugmentator import HieroglyphAugmentator
import cv2


class HieroglyphDataset(Dataset):

    #angle_sh: angulo o factor de sesgado(será *0.1)
    def __init__(self, generator_length: int, augmentator: HieroglyphAugmentator):
        self.generator_length = generator_length
        self.augmentator = augmentator

    def __len__(self):
        return self.generator_length

    def __getitem__(self, idx):
        label = idx
        image, _ = self.augmentator.augment(idx)  # Get image from augmentator

        # Ensure image is grayscale and has the correct shape
        if len(image.shape) == 2:  # If grayscale, add channel dimension
            image = image[:, :, None]  # Convert (H, W) -> (H, W, 1)
        
        # Convert to tensor and normalize
        image = torch.tensor(image, dtype=torch.float32) / 255.0  # Scale to [0,1]
        image = image.permute(2, 0, 1)  # Convert (H, W, C) -> (C, H, W) for PyTorch

        return image, label

