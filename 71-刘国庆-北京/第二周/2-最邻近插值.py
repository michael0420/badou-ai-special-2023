import cv2
import numpy as np


def function(img):
    h, w, c = img.shape
    emptyImage = np.zeros((800, 800, c), np.uint8)
    sh = 800 / h
    sw = 800 / w
    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage


img=cv2.imread("lenna.png")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("zoom",zoom)
cv2.imshow("img",img)
cv2.waitKey(0)
