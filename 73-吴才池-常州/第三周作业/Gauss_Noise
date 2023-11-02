import numpy as np
import cv2
from numpy import shape
import random
def GaussNoise(src,mu,sigma,percetage):
    Noiseimg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])

    for i in range(NoiseNum):
        #随机生成行和列
        randx=random.randint(0,src.shape[0]-1)
        randy=random.randint(0,src.shape[1]-1)

        #原像素加上高斯随机数
        Noiseimg[randx,randy]= Noiseimg[randx,randy]+random.gauss(mu,sigma)

        #防止越界
        if Noiseimg[randx,randy]<0:
            Noiseimg[randx,randy]=0
        elif Noiseimg[randx,randy]>255:
            Noiseimg[randx,randy]=255
    return Noiseimg
img=cv2.imread('lenna.png',0)
img1=GaussNoise(img,2,4,0.8)
cv2.imshow('g_',img1)
cv2.imshow('source',img)


cv2.waitKey(0)






