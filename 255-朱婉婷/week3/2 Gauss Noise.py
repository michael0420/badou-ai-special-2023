"""
高斯噪声
"""

import cv2
import numpy as np
import random

def GaussianNoise(src,means,sigma,percentage):
    NoiseImg = src.copy()
    NoiseNum = int(percentage*src.shape[0]*src.shape[1])

    for i in range(NoiseNum):
        randX = random.randint(0,src.shape[1]-1)
        randY = random.randint(0,src.shape[0]-1)
        NoiseImg[randY,randX] = NoiseImg[randY,randX]+random.gauss(means,sigma)
        #确保灰度值在[0,255]内
        if NoiseImg[randY,randX]<0:
            NoiseImg[randY,randX]=0
        elif NoiseImg[randY,randX]>255:
            NoiseImg[randY,randX]=255
    return NoiseImg

img = cv2.imread('lenna.png',0)
img_gauss = GaussianNoise(img, 2, 4, 0.8)
cv2.imshow("GaussianNoise",np.hstack([img,img_gauss]))
cv2.waitKey(0)
