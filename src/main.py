import HieroglyphCharacterGenerator
from HieroglyphDataset import HieroglyphDataset
from HieroglyphAumentator import HieroglyphAugmentator
import cv2
import sys

paths = [ 
        "./files/fonts/Noto_Sans_Egyptian_Hieroglyphs/NotoSansEgyptianHieroglyphs-Regular.ttf",
        "./files/fonts/NewGardiner/NewGardinerBMP.ttf",
            ]
    
    
ranges = [ 
    (0x00013000, 0x0001342E),
    (0x0000E000, 0x0000E42E),
        ]

def main():

    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    generator = HieroglyphCharacterGenerator.HieroglyphCharacterGenerator(paths[0], ranges[0][0], ranges[0][1], font_size=100)

    augmentator = HieroglyphAugmentator(generator)
    
    hdataset = HieroglyphDataset(generator, augmentator=augmentator, mask=struct_element, angle_sh=2)

    firsts_and_lasts = [0, 1070, 1071, 2141, 2142, 3212, 3213, 4283, 4284, 5354, 5355, 6425, 6426, 7496, 7497, 8567]#, 8568, 9638]
    randoms = [x for x in range(80,8569,1070)]
    randoms3 = [80, 1151, 2222, 3293, 4363, 5434, 6506, 7577]

    print(len(hdataset))
    for idx in firsts_and_lasts:
        img_label = hdataset.__getitem__(idx)
        (img, label) = img_label
        cv2.imwrite(f"images/img{idx}.png", img)
    print(img_label)

    print("Mostrando imagenes..")


if __name__ == "__main__":
    main()