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
    character = int(sys.argv[1])

    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    generator = HieroglyphCharacterGenerator.HieroglyphCharacterGenerator(paths[0], ranges[0][0], ranges[0][1], font_size=100)

    augmentator = HieroglyphAugmentator(generator)
    
    hdataset = HieroglyphDataset(generator, augmentator=augmentator, mask=struct_element, angle_sh=2)

    print(len(hdataset))
    img_label = hdataset.__getitem__(character)
    (img, label) = img_label
    #print(f"{img.shape}, label = {label}")
    cv2.imwrite("images/img.png", img)
    print(img_label)

    print("Mostrando imagenes..")


if __name__ == "__main__":
    main()