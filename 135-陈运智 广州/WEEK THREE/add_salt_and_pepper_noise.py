import random

import cv2


def add_salt_and_pepper_noise(src,percetage):
    NoiseImg = src
    NoiseNum = int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)
        if random.random()<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

img =cv2.imread("lenna.png",0)
img1 = add_salt_and_pepper_noise(img,0.1)
cv2.imwrite("add_salt_and_pepper_noise.png",img1)