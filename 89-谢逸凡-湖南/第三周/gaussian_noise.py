import cv2
import numpy as np
from numpy import shape
import random

def Gaussian_noise(src, simga, means, percent):
    gauss_img = src
    h, w = gauss_img.shape
    noise_number = int(percent * h * w)
    for i in range(noise_number):
        randX = random.randint(0, h - 1)
        randY = random.randint(0, w - 1)
        gauss_img[randX, randY] = gauss_img[randX, randY] + random.gauss(means, simga)
        if gauss_img[randX, randY] > 255:
            gauss_img[randX, randY] = 255
        if gauss_img[randX, randY] < 0:
            gauss_img[randX, randY] = 0
    return gauss_img

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gauss_img = Gaussian_noise(gray, 3, 3, 0.9)
cv2.imshow("gray", gray)
cv2.imshow("gauss img", gauss_img)
cv2.waitKey(0)