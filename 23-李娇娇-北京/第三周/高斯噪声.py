# -*- coding: utf-8 -*-

# @FileName: 高斯噪声.py
# @Time    : 2023/11/2 4:14 PM
# @Author  : lijiaojiao
# @Software: PyCharm

import cv2
import random


def GaussianNoise(src, mean, sigma, percetage):
    noiseImg = src
    noiseNum = int(src.shape[0] * src.shape[1] * percetage)
    for i in range(noiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，x 代表随机生成的行，y代表随机生成的列
        # random.randint生成随机整数
        # 高斯噪声图片边缘不处理，故-1
        x = random.randint(0, src.shape[0] - 1)
        y = random.randint(0, src.shape[1] - 1)
        # 原有像素灰度值上加上随机数
        noiseImg[x, y] = noiseImg[x, y] + random.gauss(mean, sigma)
        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if noiseImg[x, y] < 0:
            noiseImg[x, y] = 0
        elif noiseImg[x, y] > 255:
            noiseImg[x, y] = 255
    return noiseImg


if __name__ == '__main__':
    img = cv2.imread('../images/lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('lenna', gray)

    gaussianNoise = GaussianNoise(gray, 2, 4, 0.8)
    cv2.imshow('gaussianNoise', gaussianNoise)

    cv2.waitKey(0)



