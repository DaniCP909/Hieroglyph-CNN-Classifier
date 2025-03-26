import cv2
import sys
import random
import numpy as np
import torch

from components.HieroglyphCharacterGenerator import HieroglyphCharacterGenerator
from components.HieroglyphAugmentator import HieroglyphAugmentator
from components.HieroglyphDataset import HieroglyphDataset

paths = [ 
        "./files/fonts/Noto_Sans_Egyptian_Hieroglyphs/NotoSansEgyptianHieroglyphs-Regular.ttf",
        "./files/fonts/NewGardiner/NewGardinerBMP.ttf",
            ]
    
    
ranges = [ 
    (0x00013000, 0x0001342E),
    (0x0000E000, 0x0000E42E),
        ]

def main():

    noto_length = (ranges[0][1] - ranges[0][0])

    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    all_generators = []
    for path,hex_range in zip(paths,ranges):
        all_generators.append(HieroglyphCharacterGenerator(path, hex_range[0], hex_range[1], font_size=100))

    print(f"LONG: {len(all_generators)}")
    augmentator = HieroglyphAugmentator(all_generators, fill=False)
    
    hdataset = HieroglyphDataset(all_generators[0].getFontLength(), augmentator)

    #firsts_and_lasts = [0, 1070, 1071, 2141, 2142, 3212, 3213, 4283, 4284, 5354, 5355, 6425, 6426, 7496, 7497, 8567]#, 8568, 9638]
    #randoms = [x for x in range(80,8569,1070)]
    #randoms3 = [8556, 8557, 8558, 8559, 8560, 8561, 8562, 8563, 8564, 8565, 8566, 8567]
    random.seed(3333)
    randoms20 = [random.randint(0, len(hdataset)) for _ in range(20)]

    print(f"Longitud fuente: {noto_length}")
    dataset_len = len(hdataset)
    print(f"Longitud dataset: {dataset_len}")
    print(f"Tipo de data_len: {type(dataset_len)}")
    for i in range(5):
        augmentator.incrementSeed(i)
        for idx in randoms20:
            img_label = hdataset[idx]
            (img, label) = img_label
            # Convertir de tensor a NumPy
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()

            # Eliminar la dimensión del canal si es 1 -> De (1, H, W) a (H, W)
            if img.shape[0] == 1:
                img = img.squeeze(0)

            # Convertir a uint8
            final_img = (img * 255).astype(np.uint8)
            cv2.imwrite(f"./results/images/img{idx}_{i}.png", final_img)

    print("Mostrando imagenes..")


if __name__ == "__main__":
    main()