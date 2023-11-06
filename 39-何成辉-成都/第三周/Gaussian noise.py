# 随机生成符合正态（高斯）分布的随机数，means,sigma为两个参数
import cv2
import random

"""
@author: BraHitYQ
Gaussian noise(高斯噪声)
"""

def GaussianNoise(src, means, sigma, percetage):
        NoiseImg = src
        NoiseNum = int(percetage*src.shape[0]*src.shape[1])
        for i in range(NoiseNum):
            # 每次取一个随机点
            # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
            # random.randint生成随机整数
            # 高斯噪声图片边缘不处理，故-1

            randX = random.randint(0, src.shape[0]-1) # 随机选择行索引
            randY = random.randint(0, src.shape[1]-1) # 随机选择列索引

            # 此处在原有像素灰度值上加上随机数，生成高斯噪声值
            NoiseImg[randX, randY] = NoiseImg[randX, randY]+random.gauss(means, sigma)
            # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
            if NoiseImg[randX, randY] < 0:
                NoiseImg[randX, randY] = 0
            elif NoiseImg[randX, randY] > 255:
                NoiseImg[randX, randY] = 255
        return NoiseImg


img = cv2.imread('lenna.png',0)
img1 = GaussianNoise(img,2,4,0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('source',img2)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)


