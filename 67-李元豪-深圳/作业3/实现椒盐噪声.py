# -*- coding: utf-8 -*-  
__author__ = '李元豪 from https://www.zhilu.space'

import cv2  # 导入cv2库，cv2是OpenCV的库，OpenCV是一个开源的计算机视觉和机器学习软件库。
import random  # 导入random库，random库提供了生成随机数的功能。
import os  # 导入os库，os库提供了很多与操作系统交互的函数，例如访问文件路径、创建文件夹等。
import numpy as np  # 导入numpy库，numpy是Python的一个数学库，提供大量的数学函数以及矩阵运算的功能。


# 定义一个函数add_salt_pepper，该函数的作用是为图像添加"盐和胡椒"噪声
def add_salt_pepper(img , prob) :
    resultImg = np.zeros (img.shape , np.uint8)  # 创建一个与输入图像大小相同但所有像素值为0的新图像。
    thres = 1 - prob  # 创建阈值，prob为添加噪声的概率，那么1-prob就是没有噪声的概率。
    for i in range (img.shape[0]) :  # 遍历图像的所有行。
        for j in range (img.shape[1]) :  # 遍历图像的所有列。
            rdn = random.random ()  # 生成一个0到1之间的随机数。
            if rdn < prob :  # 如果随机数小于添加噪声的概率。
                resultImg[i][j] = 0  # 将对应像素设置为0（黑色）。
            elif rdn > thres :  # 如果随机数大于没有噪声的概率。
                resultImg[i][j] = 255  # 将对应像素设置为255（白色）。
            else :  # 如果随机数介于两者之间。
                resultImg[i][j] = img[i][j]  # 保持原始像素值不变。
    return resultImg  # 返回处理后的图像。


img1 = cv2.imread (r"D:\AI1026\a22.png")
out_img = add_salt_pepper(img1,0.06)

        # 显示处理后的图像
cv2.imshow ("img" , out_img)

        # 等待用户按键后关闭显示窗口
cv2.waitKey (0)

