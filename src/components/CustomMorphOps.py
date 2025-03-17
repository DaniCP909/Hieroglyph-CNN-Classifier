import cv2
import numpy as np

def dilate(img, mask):
    img_dilation = cv2.dilate(img, mask)
    return img_dilation


def close(img, mask, iterations):
    close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, mask, iterations=iterations)
    return close

def erode(img, mask):
    erode = cv2.dilate(img, mask)
    return erode

def rotate(img, angle):
    if angle == 0: return img

    center_of_rotation = (img.shape[0]/2,img.shape[1]/2)
    matrix = cv2.getRotationMatrix2D(center_of_rotation,angle,1)
    rotated_img = cv2.warpAffine(img,matrix,img.shape)
    return rotated_img

#transformacion de sesgado
def shear(img, sh_factor):
    if sh_factor == 0.0: return img

    (h, w) = img.shape
    matrix = np.float32([
        [1, sh_factor * 2, 0],
        [0, 1, 0]
    ])
    new_width = w + int(sh_factor * h)
    return cv2.warpAffine(img, matrix, (new_width, h))

def crop(img, padding=2):
        img_copy = img
        contours, hierarchy = cv2.findContours(img_copy,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        return_img = img_copy[max(0, y - padding): y+h + padding, max(0, x - padding): x+w + padding]
        return return_img
    
def resize_to_square(img, padding=0):
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
