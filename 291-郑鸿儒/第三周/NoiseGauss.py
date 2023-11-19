import cv2
import random


def noise_gauss(src, sigma, mean, percentage=1):
    NoiseImg = src
    NoiseNum = int(src.shape[0] * src.shape[1] * percentage)

    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[1] - 1)
        randY = random.randint(0, src.shape[0] - 1)
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(sigma, mean)

        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


img = cv2.imread('lenna.png', 0)
noise_img = noise_gauss(img, 2, 4, 0.8)
img = cv2.imread('lenna.png', 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source', img_gray)
cv2.imshow('Gaussian lenna', noise_img)
cv2.waitKey()
