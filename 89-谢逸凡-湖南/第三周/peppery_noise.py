import numpy as np
import cv2
from numpy import shape
import random

def fun1(src, percent):
    noise_img = src
    h, w = noise_img.shape
    noise_num = int(percent * h * w)
    for i in range(noise_num):
        randX = random.randint(0, h - 1)
        randY = random.randint(0, w - 1)
        
        if random.random() <= 0.5:
            noise_img[randX, randY]=0
        else:
            noise_img[randX, randY] = 255
    return noise_img

img = cv2.imread('lenna.png', 0)
img1 = fun1(img,0.15)

cv2.imwrite('lenna_PepperandSalt.png',img1)

img2 = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source', img2)
cv2.imshow('lenna_PeperandSalt',img1)
cv2.waitKey(0)


