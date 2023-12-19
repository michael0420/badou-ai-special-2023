#!/usr/bin/env python
# encoding=utf-8
import cv2
import numpy as np
import random


def GaussianNoise(src, mean, sigma, percetage=0.05):
    noise_img = src.copy()
    noise_num = int(percetage * noise_img.shape[0] * noise_img.shape[1])
    for i in range(noise_num):
        randX = random.randint(0, noise_img.shape[0] - 1)
        randY = random.randint(0, noise_img.shape[1] - 1)
        noise_img[randX, randY] = noise_img[randX, randY] + random.gauss(mean, sigma)
        if noise_img[randX, randY] < 0:
            noise_img[randX, randY] = 0
        elif noise_img[randX, randY] > 255:
            noise_img = 255
    return noise_img


img = cv2.imread("img/lenna.png", 0)
gaussian_img = GaussianNoise(img, 2, 4, 0.8)

cv2.imshow("compare", np.hstack((img, gaussian_img)))
cv2.waitKey(0)
