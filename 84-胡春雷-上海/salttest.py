import numpy as np
import cv2
from numpy import shape
import random
def  saltnoise(src,percetage):
    NoiseImg=src
    Noise_num=int(percetage*src.shape[0]*src.shape[1])
    for i in range(Noise_num):
        random.seed()
        randX=random.randint(0,src.shape[0]-1)
        random.seed()
        randY=random.randint(0,src.shape[1]-1)

        if random.random()<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

img=cv2.imread('lenna.png',0)
img1=saltnoise(img,1)

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('src',img2)
cv2.imshow('lenna_Salt',img1)
cv2.waitKey(0)
