import numpy as np
import cv2

def f(img,m):
    h,w,c = img.shape

    m1,m2 = m[0],m[1]
    s1,s2 = h/m1,w/m2
    imgg = np.zeros([m1, m2, c], np.uint8)
    for i in range(m1):
        for j in range(m2):
            x = (i+0.5)*s1-0.5
            y = (j+0.5)*s2-0.5

            x1 = int(x+0.5)
            x2 = min(x1+1,h-1)
            y1 = int(y+0.5)
            y2 = min(y1+1,w-1)

            #分两步单线性插值
            q1 = (x2-x)*img[x2,y1] + (x-x1) * img[x1,y1]
            q2 = (x2 - x) * img[x2, y2] + (x - x1) * img[x1, y2]
            imgg[i,j] = (y2 - y) * q1 + (y - y1) * q2
    return imgg



img = cv2.imread('lenna.png')
im1 = f(img,(800,700))
im2 = f(img,(250,300))

cv2.imshow('img',img)
cv2.imshow('im1',im1)
cv2.imshow('im2',im2)
cv2.waitKey(0)