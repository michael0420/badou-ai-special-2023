import cv2
import random

def PS(src,percentage):
    img = src
    imgNum = int(src.shape[0]*src.shape[1] * percentage)

    for i in range(imgNum):
        randX = random.randint(0,src.shape[0]-1) #去掉边缘
        randY = random.randint(0,src.shape[1]-1)
        randP = random.random()
        if randP <= 0.5:
            img[randX,randY] = 0 
        else:
            img[randX,randY] = 255
    return img
img = cv2.imread('lenna.png')
img0 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img = cv2.imread('lenna.png',0)
img1 = PS(img,0.2)  #只对百分之二十的部分做椒盐噪声

cv2.imshow('scource',img0)
cv2.imshow('PepperSalt',img1)
cv2.waitKey(0)