#最邻近差值算法
import cv2
img = cv2.imread('lenna.png')
h, w, channels = img.shape
import numpy as np
img_nearest = np.zeros([800, 800, channels], img.dtype)
for c in range(channels):
    for i in range(800):
        for j in range(800):
            img_nearest[i, j, c] = img[int(i*h/800+0.5), int(j*w/800+0.5), c]
cv2.imshow("image",img)
cv2.imshow("nearest interp",img_nearest)
cv2.waitKey(0)
