import random
import cv2
def GaussianNoise(src, mean, sigma,percentage):
    img = src
    imgNum = int(src.shape[0]*src.shape[1] * percentage)

    for i in range(imgNum):
        randX = random.randint(0,src.shape[0]-1) #去掉边缘
        randY = random.randint(0,src.shape[1]-1)
        img[randX,randY] = img[randX,randY] + random.gauss(mean,sigma)
        if img[randX,randY] < 0:
            img[randX,randY]
        elif img[randX,randY] >255:
            img[randX,randY] = 255
    return img

img = cv2.imread('lenna.png')
img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.imread('lenna.png',0)
img1 = GaussianNoise(img,2,4,0.9)
#cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('source',img0)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)