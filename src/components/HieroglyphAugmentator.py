import math
import cv2
import numpy as np
from typing import List
import random
from HieroglyphCharacterGenerator import HieroglyphCharacterGenerator
from CustomMorphOps import dilate, erode, close, rotate, shear, crop, resize_to_square, fill

STRUC_ELEM_SHAPE = (3,3)
MIN_ANGLE = 0
MAX_ANGLE = 4
MIN_SHEAR_FACTOR = 0.00
MAX_SHEAR_FACTOR = 0.20
MIN_ITERS = 0
MAX_ITERS = 1

class HieroglyphAugmentator:

    augmentOperations = {
        "dilate": dilate,
        "erode": erode,
        "rotate": rotate,
        "shear": shear
    }

    def __init__(self, generators: List[HieroglyphCharacterGenerator], mask=None, seed=0, fill=False):
        self.generators = generators
        self.mask = mask
        self.origin_seed=seed
        self.fill = fill
        print(f"........... SEED(saved not stablished): {self.origin_seed}")

    def incrementSeed(self, epoch: int):
        new_seed = self.origin_seed + epoch
        random.seed(new_seed)
        np.random.seed(new_seed)
        print(f"........... SEED(update): {new_seed}")

    def incrementTestSeed(self):
        return self.origin_seed

    def augment(self, idx, seed: int=None):

        morphValues = self._initMorphValues(struct_element=self.mask)

        selected_generator = self.generators[morphValues['generator']]
        raw_img = selected_generator.getImageByLabel(idx)

        if(self.fill): raw_img = fill(raw_img)

        img = resize_to_square(crop(raw_img, 0), 10)

        if morphValues['close']:
            distorted_image = close(img, self.mask, morphValues['nCloseIters'])
        else:
            distorted_image = dilate(img, mask=self.mask)
        
        rotated = rotate(distorted_image, morphValues['angle'])

        sheared = shear(rotated, morphValues['sh_factor'])

        result = crop(sheared, 0)
        result = cv2.resize(resize_to_square(result, 10), (100, 100))

        return (result, seed)

    
    def _initMorphValues(self, struct_element=None):

        if struct_element is None:  # 3x3 elem with random values but center = 1
            strucElementShape = (3,3)
            strucElement = np.random.randint(0, 2, strucElementShape)
            strucElement[strucElementShape[0] // 2, strucElementShape[1] // 2] = 1
        else:
            strucElement = struct_element

        #print(len(self.generators))

        return {
            'generator': random.randint(0, (len(self.generators) - 1)),
            'strucElement': strucElement,
            'close': bool(random.getrandbits(1)),
            'nCloseIters': random.randint(MIN_ITERS, MAX_ITERS),
            'angle': random.randint(MIN_ANGLE ,MAX_ANGLE),
            'sh_factor': round(random.uniform(MIN_SHEAR_FACTOR, MAX_SHEAR_FACTOR), 1)
        }

