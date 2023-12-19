import cv2
import numpy as np
from numpy import shape
import random

'''
:param src: 要加噪声的图片
:param means: 高斯噪声的均值
:param sigma: 高斯噪声的标准差
:param percetage: 需要加噪声的像素占总像素的比例
'''
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    # 计算需要加噪声的总像素数
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):

        # random.randint生成随机整数
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[0]-1)

        # 在原有像素灰度值上加上随机数
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)

        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if NoiseImg[randX,randY]<0:
            NoiseImg[randX,randY]=0
        elif NoiseImg[randX,randY]>255:
            NoiseImg[randX,randY]=255
    return NoiseImg

img = cv2.imread('lenna.png',0)
img1 = GaussianNoise(img,2,4,0.9)
img = cv2.imread('lenna.png',1)
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow('GaussianNoise',img1)
cv2.imshow('src',img2)
cv2.waitKey(0)