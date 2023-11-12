import random
import numpy as np
import cv2

def salt_and_pepper_noise(img, snr):
    """
    <2023.9.13> version:1.0
    给输入图像加椒盐噪声
    :param:img,输入图像
    :param:snr,指定信噪比,信号和噪声所占比例，取值在[0,1]之间
    :return:noise_img,加噪后的图像
    """
    noise_img = img
    src_w, src_h = img.shape  # 获取图像参数时，先宽后高
    noise_nums = int(snr*src_h*src_w)  # 要加噪的像素数目
    for i in range(noise_nums):
        dst_x = random.randint(0, src_w - 1)  # 生成随机坐标
        dst_y = random.randint(0, src_h - 1)
        # random.random()用于生成一个 [0, 1) 范围内的随机浮点数
        if random.random() <= 0.5:  # 有一半的概率丢失后为0/255
            noise_img[dst_x, dst_y] = 0
        else:
            noise_img[dst_x, dst_y] = 255
    return noise_img

snr = 0.2
img = cv2.imread("D:\\subject_learning\\cv_learn\\project\\lenna.jpg", 0)
cv2.imshow('img', img)
noise_img = salt_and_pepper_noise(img, snr)
cv2.imshow('noise_img', noise_img)
cv2.waitKey(0)