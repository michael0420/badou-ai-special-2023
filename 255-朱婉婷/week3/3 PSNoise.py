"""
椒盐噪声
"""
import cv2
import numpy as np
import random

def PS_noise(src,percentage):
    NoiseImg = src.copy()
    NoiseNum = int(percentage*src.shape[0]*src.shape[1])

    for i in range(NoiseNum):
        randX = random.randint(0,src.shape[1]-1)
        randY = random.randint(0,src.shape[0]-1)
        #椒盐噪声改变
        if random.random()<=0.5:
            NoiseImg[randY,randX]=0
        else:
            NoiseImg[randY,randX]=255 
    return NoiseImg

img = cv2.imread('lenna.png',0)
cv2.imshow('img',img)

img_PSNoise = PS_noise(img, 0.2)
#cv2.imshow('Pepper and Salt Noise',np.hstack([img,img_PSNoise]))
cv2.imshow('Pepper and Salt Noise',img_PSNoise)
cv2.waitKey(0)
