import cv2
import random
import numpy as np


def function(src,percetage):
    noiseImg = src
    noiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0,src.shape[0] - 1)
        randY = random.randint(0,src.shape[1] - 1)
        if random.random() <= 0.5:
            noiseImg[randX,randY] = 0
        else:
            noiseImg[randX, randY] = 255
    return noiseImg

if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img1 = function(img,0.6)
    cv2.imshow('gary', img2)
    cv2.imshow('jiaoyan', img1)
    cv2.waitKey(0)
