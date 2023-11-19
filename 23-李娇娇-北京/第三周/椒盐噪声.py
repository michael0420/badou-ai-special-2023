# -*- coding: utf-8 -*-

# @FileName: 椒盐噪声.py
# @Time    : 2023/11/2 4:43 PM
# @Author  : lijiaojiao
# @Software: PyCharm

import cv2
import random


def SaltAndPepperNoise(src, percetage):
    noiseImg = src.copy()
    noiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(noiseNum):
        # 随机取一个像素点
        x = random.randint(0, src.shape[0] - 1)
        y = random.randint(0, src.shape[1] - 1)

        # 椒盐噪声
        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() < 0.5:
            noiseImg[x, y] = 0
        else:
            noiseImg[x, y] = 255
    return noiseImg


if __name__ == '__main__':
    img = cv2.imread('../images/lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)

    noise = SaltAndPepperNoise(gray, 0.1)
    cv2.imshow('noise', noise)

    cv2.waitKey(0)

