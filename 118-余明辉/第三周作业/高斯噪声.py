import numpy as np
import cv2
from numpy import shape
import random

def gs(img,u ,sigma,p):
    num = img.shape[0]*img.shape[1]
    nump = int(num * p)
    im = img
    for i in range(nump):
        x = random.randint(0,img.shape[0]-1)
        y = random.randint(0,img.shape[1]-1)

        im[x,y] = img[x,y] + random.gauss(u,sigma)

        if im[x,y] <= 0:
            im[x, y] = 0
        if im[x,y] > 255:
            im[x, y] = 255
    return im

img = cv2.imread('lenna.png',0)
im = gs(img,0,1,0.7)
img = cv2.imread('lenna.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('img & im',np.hstack([img,im]))

cv2.waitKey(0)