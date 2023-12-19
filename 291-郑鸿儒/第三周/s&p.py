#!/usr/bin/env python
# encoding=utf-8
import cv2
import random
import numpy as np


def sp_noise(src, percetage=0.05, chans=0):
    noise_img = src.copy()
    noise_num = int(percetage * noise_img.shape[0] * noise_img.shape[1])
    for i in range(noise_num):
        randX = random.randint(0, noise_img.shape[0] - 1)
        randY = random.randint(0, noise_img.shape[1] - 1)
        target = 255 if random.random() > 0.5 else 0
        if chans:
            noise_img[randX, randY, random.randint(0, chans - 1)] = target
        else:
            noise_img[randX, randY] = target
    return noise_img


img = cv2.imread("img/lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sp_img = sp_noise(gray)
sp_orgin_img = sp_noise(img, chans=3)
cv2.imshow("compare", np.hstack((gray, sp_img)))
cv2.imshow("channels compare", np.hstack((img, sp_orgin_img)))
cv2.waitKey(0)
cv2.destroyAllWindows()
