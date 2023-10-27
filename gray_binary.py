#灰度 二值化
import cv2
import matplotlib.pyplot as plt
import numpy as np


def gray(img):
    h,w=img.shape[:2]
    image_gray = np.zeros([h,w],img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i,j]
            image_gray[i,j]=m[0]*0.11+m[1]*0.59+m[2]*0.3
    return image_gray
img = cv2.imread('lenna.png')
img_gray = gray(img)
cv2.imshow("", img_gray)


#二值化
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if(img_gray[i,j]<=0.5):
            img_gray[i, j]=0
        else:
            img_gray[i,j]=1

print(img_gray)
print(img_gray.shape)
plt.subplot(223)
plt.imshow( img_gray, cmap="gray")
plt.show()
