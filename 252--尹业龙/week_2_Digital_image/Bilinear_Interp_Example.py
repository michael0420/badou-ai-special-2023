"""
双线性插值算法测试用例
Bilinear_Interp_Example
data:2023.9.8
"""

import cv2
import numpy as np
from Bilinear_Interp import bilinear_interp

width = 700
height = 700
out_dim = (width, height)
img = cv2.imread("D:\\subject_learning\\cv_learn\\project\\lenna.jpg")
empty_img = bilinear_interp(img, out_dim)
cv2.imshow('img', img)
cv2.imshow('empty_img', empty_img)
cv2.waitKey(0)  # 在程序中添加一个等待键盘输入的循环，等待用户按下任意键后关闭图像窗口;不加会瞬间自动关闭打开的图像

