import cv2 
import numpy as np


def function(img):
    h,w,c = img.shape
    target_img = np.zeros([1000,1000,c],np.uint8)
    sh = 1000/h
    sw = 1000/w
    for i in range(1000):
        for j in range(1000):
            x = int(i/sh + 0.5) # int只会向下取整 所以+0.5为了区分取整的方法
            y = int(j/sw + 0.5)
            target_img[i,j] = img[x,y]
    return target_img

img = cv2.imread('lenna.png')
target_img = function(img)
cv2.imshow("nearest interpolation",target_img)
cv2.imshow("image",img)
cv2.waitKey(0)
