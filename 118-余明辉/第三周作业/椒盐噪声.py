import numpy as np
import cv2
from numpy import shape
import random

def jiaoyan(img,p):
    num = img.shape[0]*img.shape[1]
    nump = int(num * p)
    im = img
    for i in range(nump):
        x = random.randint(0,img.shape[0]-1)
        y = random.randint(0,img.shape[1]-1)

        s = random.random()
        if s>0.5:
            im[x,y] = 0
        else:
            im[x,y] = 255


    return im

img = cv2.imread('lenna.png',0)
im = jiaoyan(img,0.1)
img = cv2.imread('lenna.png',0)
cv2.imshow('img & im',np.hstack([img,im]))

cv2.waitKey(0)