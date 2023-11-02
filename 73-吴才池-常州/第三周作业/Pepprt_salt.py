import numpy as np
import cv2
from numpy import shape
import random
def func(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])

    for i in range(NoiseNum):

        randx=random.randint(0,src.shape[0]-1)
        randy=random.randint(0,src.shape[1]-1)

        if random.random()<=0.5:
            NoiseImg[randx,randy]=0
        else:
            NoiseImg[randx,randy]=255
    return NoiseImg

img=cv2.imread("lenna.png",0)
cv2.imshow("source",img)
img1=func(img,0.2)
cv2.imshow("pepper",img1)
cv2.waitKey(0)
