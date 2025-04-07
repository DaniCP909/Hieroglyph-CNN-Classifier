import cv2
import sys
import random
import numpy as np
import torch

from components.HieroglyphCharacterGenerator import HieroglyphCharacterGenerator
from components.HieroglyphAugmentator import HieroglyphAugmentator
from components.HieroglyphDataset import HieroglyphDataset


path_short = [
    "./files/fonts/egyptian-hieroglyphs-silhouette/EgyptianHieroglyphsSilhouet.otf"
]
short_font_tags = [33,36,37,40,41,43,45,49,50,51,52,53,54,55,56,57,64,
                   65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,81,82,
                   83,84,85,86,87,88,89,90,97,98,99,100,101,102,103,104,
                   105,106,107,108,109,110,111,112,113,114,115,116,117,
                   118,119,120,121,122,162,163,165]
print(f"Tags list len: {len(short_font_tags)}")

range_short = (0, len(short_font_tags) - 1)
print(f"Range: {range_short}")

def main():

    short_generator = HieroglyphCharacterGenerator(path_short[0], range_short[0], range_short[1], font_size=100, short_font=True)

    augmentator = HieroglyphAugmentator([short_generator])
    
    hdataset = HieroglyphDataset(short_generator.getFontRange, augmentator)

    #firsts_and_lasts = [0, 1070, 1071, 2141, 2142, 3212, 3213, 4283, 4284, 5354, 5355, 6425, 6426, 7496, 7497, 8567]#, 8568, 9638]
    #randoms = [x for x in range(80,8569,1070)]
    #randoms3 = [8556, 8557, 8558, 8559, 8560, 8561, 8562, 8563, 8564, 8565, 8566, 8567]
    random.seed(3333)
    randoms20 = [x for x in range_short]
    for i in range(5):
        for idx in range(range_short[1] + 1):
            img_label = hdataset[idx]
            (img, label) = img_label
            # Convertir de tensor a NumPy
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()

            # Eliminar la dimensión del canal si es 1 -> De (1, H, W) a (H, W)
            if img.shape[2] == 1:
                img = img[:, :, 0]

            # Convertir a uint8
            final_img = (img * 255).astype(np.uint8)
            cv2.imwrite(f"./results/images/img{idx}_{i}.png", final_img)

    print("Mostrando imagenes..")


if __name__ == "__main__":
    main()