'''
处理高斯噪声的步骤：
    1、输入参数sigma和mean
    2、生成高斯随机数
    3、根据输入像素计算出输出像素
    4、重新将像素缩放在【0,255】之间
    5、循环所有像素
    6、输出图像
'''
import matplotlib.pyplot as plt                                 #导入matlotlib模组，方便后续打印观察
import numpy as np
import cv2
from numpy import shape
import random
def GaussianNoise(src,means,sigma,percetage):                   #
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)

        if  NoiseImg[randX, randY]<0:
            NoiseImg[randX, randY]=0
        elif NoiseImg[randX, randY]>255:
            NoiseImg[randX, randY]=255

    return NoiseImg

img = cv2.imread('lenna.png',0)
img1 = GaussianNoise(img,2,4,0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            #还可另外保持文件：cv2.imwrite('lenna_GaussianNoise.png',img1)

plt.subplot(121)
plt.imshow(img1,cmap='gray')                            #camp='gray'在模组打印中
plt.title("src img")

plt.subplot(122)
plt.imshow(img2,cmap='gray')
plt.title("GaussianNoise")
plt.show()

