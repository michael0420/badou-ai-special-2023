from skimage.color import  rgb2gray
import  numpy as np
import  matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread('D:/bilibili/buyaoshan.jpg')
h,w = img.shape[:2]
plt.show()
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)

plt.subplot(221)
plt.imshow(img)
plt.title("original image")
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')
plt.title("grayscale image")
plt.show()
