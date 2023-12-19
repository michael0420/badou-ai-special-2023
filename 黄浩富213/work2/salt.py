import numpy as np
import cv2
import random

img = cv2.imread('lenna.png')
gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
salt = gary
cv2.imshow('gary', gary)
cv2.imshow('original', img)
noise_num = int(img.shape[0] * img.shape[1] * 0.1)
print(noise_num)
for i in range(noise_num):
    randx = random.randint(0, img.shape[1]-1)
    randy = random.randint(0, img.shape[0]-1)
    if random.random() <= 0.5:
        salt[randy, randx] = 0
    else:
        salt[randy, randx] = 255


cv2.imshow('gauss', salt)
cv2.waitKey(0)