import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image

img = cv2.imread('lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = img.shape[:2]
img_binary = np.zeros([h, w], img.dtype)

for i in range(h):
    for j in range(w):
        num = img[i, j] / 255
        if num  >= 0.5:
            img_binary[i, j] = 255
        else:
            img_binary[i, j] = 0

cv2.imshow('first', img_binary)

cv2.waitKey()

#second
img1 = cv2.imread('lenna.png')
img_gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img_binary1 = np.where(img_gray1  >= 0.5, 1, 0)

plt.imshow(img_binary, cmap='gray')
plt.show()
