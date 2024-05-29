import math
import cv2

class HieroglyphAugmentator:

    def __init__(self, angle=None):
        self.angle = angle

    def augment(self, img):
        return self.rotate(img, self.angle)

    
    def rotate(self, img, angle):
        center_of_rotation = (img.shape[0]/2,img.shape[1]/2)
        matrix = cv2.getRotationMatrix2D(center_of_rotation,angle,1)
        cateto_x = int(math.sin(angle*math.pi/180) * img.shape[0]/2)
        cateto_y = int(math.sin(angle*math.pi/180) * img.shape[1]/2)
        matrix[0,2] = matrix[0,2] + cateto_x
        matrix[1,2] = matrix[1,2] + cateto_y
        rotated_img_size = (img.shape[0]+2*cateto_y,img.shape[1]+2*cateto_x)
        rotated_img = cv2.warpAffine(img,matrix,rotated_img_size)
        return rotated_img
