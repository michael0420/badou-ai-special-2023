import numpy as np
import cv2
from numpy import shape
import random
def fun1(src,percetage):
    NoiseImg=src
    # 计算需要加噪声的总像素数
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 椒盐噪声图片边缘不处理，故-1
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)

        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


img = cv2.imread('lenna.png')
img1 = cv2.imread('lenna.png',0)

cv2.imshow('Src',img)
cv2.imshow('Gray',img1)

# 加椒盐噪声
img2 = fun1(img,0.1)
img3 = fun1(img1,0.1)



cv2.imshow('Src_PepperandSalt',img2)
cv2.imshow('Gray_PepperandSalt',img3)
cv2.waitKey(0)