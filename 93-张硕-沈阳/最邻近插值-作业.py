import cv2
import numpy as np

def interpolation(img):
    height,width,channels = img.shape
    emptyImage = np.zeros((800,800,channels),np.uint8)
    sh = 800/height
    sw = 800/width
    for c in range(channels):
        for i in range(800):
            for j in range(800):
                x = int(i/sh + 0.5)
                y = int(j/sw + 0.5)
                emptyImage[i,j,c] = img[x,y,c]

    return emptyImage

img = cv2.imread('lenna.png')
zoom = interpolation(img)
print(zoom)
print(zoom.shape)

cv2.imshow('nearest interp', zoom)
cv2.waitKey(0)
