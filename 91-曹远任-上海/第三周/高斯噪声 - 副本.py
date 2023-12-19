import cv2
import random

import numpy as np


def GaussianNoise(src, means, sigma, percetage):
    h, w = src.shape
    result = np.zeros((h, w), dtype=src.dtype)
    for i in range(h):
        for j in range(w):
            if random.random() < percetage:
                result[i][j] = src[i][j] + random.gauss(means, sigma)
                if result[i][j] > 255:
                    result[i][j] = 255
                if result[i][j] < 0:
                    result[i][j] = 0
            else:
                result[i][j] = src[i][j]
    return result


img = cv2.imread('lenna.png', 0)
img1 = GaussianNoise(img, 0, 20, 0.8)
cv2.imshow('source', img)
cv2.imshow('lenna_GaussianNoise', img1)
cv2.waitKey(0)
