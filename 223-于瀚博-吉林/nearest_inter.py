import cv2
import numpy as np
def functional(img):
    height,width,channel=img.shape
    emptyimg=np.zeros((800,800,channel),np.uint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/sh)
            y=int(j/sw)
            emptyimg[i,j]=img[x,y]
    return emptyimg

img=cv2.imread('lenna.png')
zoom=functional(img)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)