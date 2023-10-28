import cv2
import numpy as np
def function(img):
    height,width,channels =img.shape
    newImg=np.zeros((1000,1000,channels),np.uint8)
    sh=1000/height
    sw=1000/width
    for i in range(1000):
        for j in range(1000):
            x=int(i/sh+0.5)
            y=int(j/sw+0.5)
            newImg[i,j]=img[x,y]
    return newImg
img=cv2.imread("scenery.png")
nearest=function(img)
cv2.imshow('naerest',nearest)
print('邻近插值',nearest)
cv2.imshow('img',img)
print('原始图像',img)
cv2.waitKey(0)