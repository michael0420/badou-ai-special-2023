import numpy as np
import cv2
import random

img = cv2.imread('lenna.png')
gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gauss = gary
noise_num = int(img.shape[0] * img.shape[1] * 0.8)
print(noise_num)
for i in range(noise_num):
    randx = random.randint(0, img.shape[1]-1)
    randy = random.randint(0, img.shape[0]-1)
    gauss[randy, randx] = gauss[randy, randx] + random.gauss(2,8)
    if gauss[randy, randx] > 255:
        gauss[randy, randx] = 255
    if gauss[randy, randx] < 0:
        gauss[randy, randx] = 0

cv2.imshow('gary', gary)
cv2.imshow('original', img)
cv2.imshow('gauss', gauss)
cv2.waitKey(0)