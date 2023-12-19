import random
import cv2
import numpy as np

def gussian_noise(img, sigma, mean, percentage):
    """
    <2023.9.13> version:1.0
    给输入图像加上高斯噪声
    :param:img,输入图像
    :param:sigma,标准差
    :param:mean,均值
    :param:percentage,噪声点占比,[0,1]之间的小数
    :return:noise_img,加噪后的图像
    """
    src_w, src_h = img.shape  # 获取图像参数时，先宽后高
    noise_nums = int(percentage*src_w*src_h)  # 根据图像总像素数和噪声占比获取图像中的噪声点数
    noise_img = np.zeros((src_w, src_h), np.uint8)
    for i in range(noise_nums):
        # 随机确定噪声点的位置，-1是因为图像的边缘无法取到【python从0开始，图像像素最后一个点假如是512取不到】
        dst_x = random.randint(0, src_w - 1)
        dst_y = random.randint(0, src_h - 1)
        noise_img[dst_x, dst_y] = img[dst_x, dst_y] + random.gauss(mean, sigma)
        # 像素点值的缩放
        # noise_img[dst_x, dst_y]是一个1x3的数组，3个通道所以有3个像素值，这是同时对图像的3个通道进行加噪
        # .any()函数是判断noise_img[dst_x, dst_y]中的每一个元素
        # reference: https://blog.csdn.net/u011699626/article/details/132422541
        if noise_img[dst_x, dst_y] < 0:
            noise_img[dst_x, dst_y] = 0
        elif noise_img[dst_x, dst_y] > 255:
            noise_img[dst_x, dst_y] = 255
    return noise_img

