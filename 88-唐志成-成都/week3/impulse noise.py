import cv2
import random


def impuseNoise(src, percetage):  # (原图， 百分比)
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        # 随机（x,y）
        randX = random.randint(0, src.shape[0] - 1)  # 不考虑边缘
        randY = random.randint(0, src.shape[1] - 1)

        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


img = cv2.imread('lenna.png', 0)
img1 = impuseNoise(img, 0.2)

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source', img2)
cv2.imshow('lenna_PepperandSalt', img1)
cv2.waitKey(0)
