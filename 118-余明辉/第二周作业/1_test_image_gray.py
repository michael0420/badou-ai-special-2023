#灰度化
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray

img = cv2.imread('lenna.png')
plt.subplot(221)
plt.imshow(img)
h,w = img.shape[:2]
print(img.shape)
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        im = img[i,j]
        img_gray[i,j] = int(im[0]*0.11 + im[1]*0.59 + im[2]*0.3)
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')

img_gray2 = rgb2gray(img)
plt.subplot(223)
plt.imshow(img_gray2,cmap='gray')
#二值化
img_binary = np.where(img_gray >= 112,1,0)
plt.subplot(224)
plt.imshow(img_binary,cmap='gray')
plt.show()