import numpy as np
import cv2
import random
def  SaltAndPepper(src,percetage):
    NoiseImg=cv2.copyTo(src=src,mask=None)
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if random.random()<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

img=cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE)
img1=SaltAndPepper(img,0.1)
cv2.imshow('compare',np.hstack([img1,img]))
cv2.waitKey(0)

