import cv2
import numpy as np

'''
最近邻插值法
'''


def function(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    sh = 800/height
    sw = 800/width

    for i in range(800):
        for j in range(800):
            x = int(i/sh)
            y = int(j/sw)
            emptyImage[i,j] = img[x,y]

    return emptyImage


img = cv2.imread('img.png')
zoom = function(img)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)
