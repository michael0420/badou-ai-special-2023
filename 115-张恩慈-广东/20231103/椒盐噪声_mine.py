# 作业3： 椒盐噪声

import random

import cv2

img = cv2.imread("lenna.png")
dst = img.copy()
height, width, channels = img.shape
# 1、输入信噪比 SNR（直接影响噪点数量）
percent = 0.1
# 2、计算噪点数
noise = int(percent * height * width)
for channel in range(channels):
    for i in range(noise):
        # 3、取任意随机点([0, width - 1], [0, height - 1])，椒盐噪声图片 边缘 不做处理
        rand_x = random.randint(0, width - 1)
        rand_y = random.randint(0, height - 1)
        # 4、生成一个 [0, 1] 的随机数
        rand = random.uniform(0, 1)
        # 5、若随机数小于0.5则置为0，大于等于0.5则置为255
        if rand < 0.5:
            dst[rand_x, rand_y, channel] = 0
        else:
            dst[rand_x, rand_y, channel] = 255
cv2.imshow('origin', img)
cv2.imshow('salt and pepper noise', dst)
cv2.waitKey(0)
