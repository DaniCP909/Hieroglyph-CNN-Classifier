import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    img = cv2.imread("./results/images/img3_5.png")
    gray = img[:, :, 1] #todos tienen el mismo valor

    h,w = gray.shape

    img_red = np.zeros((h,w,3), dtype=np.uint8)
    img_green = np.zeros((h,w,3), dtype=np.uint8)

    img_red[:,:,2] = gray
    img_green[:,:,1] = gray

    cv2.imwrite('./files/testing_code/my_examples1.png', img_red)
    cv2.imwrite('./files/testing_code/my_examples2.png', img_green)



if __name__=="__main__":
    main()