import math
import cv2
import numpy as np
from HieroglyphCharacterGenerator import HieroglyphCharacterGenerator

class HieroglyphAugmentator:

                    # dilate, close
    morph_augments = [False, False]

                    # rotate, shear, rotate+shear
    geometric_augments = [False, False, False, False]

    def __init__(self, generator: HieroglyphCharacterGenerator):
        self.generator_len = generator.getFontLength()

    def augment(self, img, idx, mask, angle_sh, iterations):
        stage = int(idx / self.generator_len)
        m_op = int(stage / len(self.geometric_augments))
        g_op = stage % len(self.geometric_augments)
        print(f"[{stage}]: {m_op}, {g_op}")
        self.morph_augments[m_op] = True
        self.geometric_augments[g_op] = True
        label = idx % self.generator_len

        result = self.crop(img, 0)
        result = self.resize_to_square(result, 10)

        if(self.morph_augments[0] and mask is not None):
            result = self.dilate(result, mask)
        if(self.morph_augments[1] and mask is not None):
            result = self.close(result, mask, iterations)
        
        result = self.crop(result, 0)
        result = self.resize_to_square(result, 10)

        if(self.geometric_augments[1] or self.geometric_augments[3]):
            result = self.rotate(result, 2 * angle_sh)
        if(self.geometric_augments[2] or self.geometric_augments[3]):
            result = self.shear(result, angle_sh * 0.1)

        result = self.crop(result, 0)
        result = self.resize_to_square(result, 10)

        self.morph_augments[m_op] = False
        self.geometric_augments[g_op] = False

        return result


    def dilate(self, img, mask):
        img_dilation = cv2.dilate(img, mask)
        return img_dilation
    
    def open(self, img, mask):
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, mask, )
        return open

    def close(self, img, mask, iterations):
        close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, mask, iterations=iterations)
        return close
    
    def rotate(self, img, angle):
        center_of_rotation = (img.shape[0]/2,img.shape[1]/2)
        matrix = cv2.getRotationMatrix2D(center_of_rotation,angle,1)
        rotated_img = cv2.warpAffine(img,matrix,img.shape)
        return rotated_img

    #transformacion de sesgado
    def shear(self, img, sh_factor):
        (h, w) = img.shape
        matrix = np.float32([
            [1, sh_factor * 2, 0],
            [0, 1, 0]
        ])
        new_width = w + int(sh_factor * h)
        return cv2.warpAffine(img, matrix, (new_width, h))
    
    def crop(self, img, padding=2):
        img_copy = img
        contours, hierarchy = cv2.findContours(img_copy,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        return_img = img_copy[max(0, y - padding): y+h + padding, max(0, x - padding): x+w + padding]
        return return_img
    
    def resize_to_square(self, img, padding=0):
        new_image = img
        h, w = new_image.shape
        max_dim = max(h, w)
        min_dim = min(h, w)

        black_img = np.zeros((max_dim + (padding * 2), max_dim + (padding * 2)), dtype=np.uint8)

        offset = int((black_img.shape[0] - min_dim) / 2)
        if(max_dim == h):
            black_img[padding:max_dim + padding, offset:offset+min_dim] = new_image
        else:
            black_img[offset:offset+min_dim, padding:max_dim + padding] = new_image
        return black_img
    
    def getAugmentatorLength(self):
        return len(self.morph_augments) * len(self.geometric_augments)
