import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image

#使用遍历实现
img = cv2.imread('lenna.png')
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i, j]=int(m[0]*0.3 + m[1]*0.59 + m[2]*0.11)

cv2.imshow('first-gray', img_gray)

#
img1 = cv2.imread('lenna.png')
img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #或者gbr2rgb后，使用rgb2gray
cv2.imshow('second-gray', img_gray1)
cv2.waitKey(0)