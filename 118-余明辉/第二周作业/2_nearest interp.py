import cv2
import numpy as np

def f(img,m):
    m = int(m)
    h,w,c = img.shape
    igg = np.zeros((800,800,c),np.uint8)
    for i in range(m):
        for j in range(m):
            imgi = int(i/m*h + 0.5)
            imgj = int(j/m*w + 0.5)
            igg[i,j] = img[imgi,imgj]
    return igg



img = cv2.imread('lenna.png')
img1 = f(img,800)#放大
img2 = f(img,300)#缩小
cv2.imshow('img',img)
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.waitKey(0)