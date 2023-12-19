# 作业2： 高斯噪声
import random

import cv2

img = cv2.imread("lenna.png")
dst = img.copy()
height, width, channels = img.shape
# 1、输入参数 sigma 和 mean 以及分布噪点的百分比（直接影响噪点数量和分布）
mean = 0
sigma = 10
percent = 0.8
# 2、计算噪点数
noise = int(percent * height * width)
for channel in range(channels):
    for i in range(noise):
        # 3、取任意随机点([0, width - 1], [0, height - 1])，高斯噪声的图片 边缘 不做处理
        rand_x = random.randint(0, width - 1)
        rand_y = random.randint(0, height - 1)
        # 4、生成高斯随机数
        num = random.gauss(mean, sigma)
        # 5、在原像素值基础上 加上 随机数
        dst[rand_x, rand_y, channel] = img[rand_x, rand_y, channel] + num
        # 6、若像素值小于0则置为0，大于255则置为255
        if dst[rand_x, rand_y, channel] < 0:
            dst[rand_x, rand_y, channel] = 0
        elif dst[rand_x, rand_y, channel] > 255:
            dst[rand_x, rand_y, channel] = 255
cv2.imshow('origin', img)
cv2.imshow('gauss', dst)
cv2.waitKey(0)
