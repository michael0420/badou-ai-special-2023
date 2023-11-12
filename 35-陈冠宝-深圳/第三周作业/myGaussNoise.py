import cv2
import numpy as np
import random
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=cv2.copyTo(src=src,mask=None)
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        if  NoiseImg[randX, randY]< 0:
            NoiseImg[randX, randY]=0
        elif NoiseImg[randX, randY]>255:
            NoiseImg[randX, randY]=255
    return NoiseImg
img = cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE)
img1 = GaussianNoise(img,2,4,0.8)
cv2.imshow('compare',np.hstack([img1,img]))
cv2.waitKey(0)
