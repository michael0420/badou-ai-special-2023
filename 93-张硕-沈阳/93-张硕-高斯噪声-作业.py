import numpy as np
import cv2
from numpy import shape
import random

def Gauss(src, means, sigma, percetage):
    NoiseImg = src.copy()
    NoiseNum = int(src.shape[0] * src.shape[1] * percetage)

    randX = random.randint(0, src.shape[0] -1 )
    randY = random.randint(0, src.shape[1] - 1)
    NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
    if NoiseImg[randX, randY] < 0:
        NoiseImg[randX, randY] = 0
    elif NoiseImg[randX, randY] > 255:
        NoiseImg[randX, randY] = 255
    return NoiseImg

img = cv2.imread('lenna.png', 0)
img1 = Gauss(img, 2, 4, 1)
cv2.imshow('vs', np.hstack([img, img1]))
cv2.waitKey(0)
