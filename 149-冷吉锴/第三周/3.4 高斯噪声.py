import cv2
import random
import numpy as np


def GaussianNoise(src, means, sigma, percetage):
    noiseImg = src  # 复制一下带噪声的图片
    noiseNum = int(percetage * src.shape[0] * src.shape[1])  # 算一下噪声的数量
    for i in range(noiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY 代表随机生成的列
        # random.randint 生成随机整数
        # 高斯噪声图片边缘不处理，故 -1
        randX = random.randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)
        # 此处在原有像素灰度值上加上随机数
        noiseImg[randX, randY] = noiseImg[randX, randY] + random.gauss(means, sigma)
        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if noiseImg[randX, randY] < 0:
            noiseImg[randX, randY] = 0
        elif noiseImg[randX, randY] > 255:
            noiseImg[randX, randY] = 255
    return noiseImg


img = cv2.imread("lenna.png", 0)  # flags=0 将图像调整为单通道的灰度图像
img1 = GaussianNoise(img, 2, 4, 0.8)  # 0.8是添加高斯噪声的百分比
img = cv2.imread("lenna.png")
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("source", img2)
cv2.imshow("lenna_GaussianNoise", img1)
cv2.waitKey(0)
