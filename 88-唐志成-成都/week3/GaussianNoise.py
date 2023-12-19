import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy import shape
import random


# -----------------------------------------原理----------------------------------------#
def GaussianNoise(src, means, sigma, percetage):
    NoiseImg = src  # 高斯图片
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 高斯噪声图片边缘不处理，故-1
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        # 此处在原有像素灰度值上加上随机数
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


#  -------------------------------------------------原理----------------------------------------#

#  彩色图片高斯噪声
image = cv2.imread('lenna.png')
height, width, channels = image.shape
# 均值，方差
mean = 2
sigma = 4
# 生成高斯噪声
noise = np.random.normal(mean, sigma, (height, width, channels))  # 生成彩色图片大小，3通道的高斯随机点
# 将噪声添加到图像
noisy_image = image + noise  # 将原图和高斯随机点图叠加
# 将像素值限制在0到255的范围内
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# 图片显示
img = cv2.imread('lenna.png', 0)  # 以灰度图模式读图
img1 = GaussianNoise(img, 2, 4, 0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('lenna_GaussianNoise.png', img1) 将图片保存到电脑中

cv2.imshow('lenna.png-RGB', img)
cv2.imshow('lenna.png', img2)
cv2.imshow('lenna_GaussianNoise', img1)
cv2.imshow('noisy_image', noisy_image)
cv2.waitKey(0)
