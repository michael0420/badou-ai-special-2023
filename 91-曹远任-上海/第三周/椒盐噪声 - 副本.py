import cv2
import random

import numpy as np


def PSNoise(src, percetage):
    h, w = src.shape
    result = np.zeros((h, w), dtype=src.dtype)
    for i in range(h):
        for j in range(w):
            if random.random() < percetage:
                result[i][j] = 255 if random.random() > 0.5 else 0
            else:
                result[i][j] = src[i][j]
    return result


img = cv2.imread('lenna.png', 0)
img1 = PSNoise(img, 0.1)
cv2.imshow('source', img)
cv2.imshow('lenna_PSNoise', img1)
cv2.waitKey(0)
