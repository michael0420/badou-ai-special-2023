"""
最邻近插值算法测试用例
Nearest_Interp_Example
data:2023.9.7
"""
import cv2
import numpy as np
from Nearest_Interp_2 import nearest_interp

height = 400
width = 400
out_dim = (height,width)
img = cv2.imread("D:\\subject_learning\\cv_learn\\project\\lenna.jpg")
empty_img = nearest_interp(img, out_dim)
# empty_img = cv2.resize(img, (height, width),interpolation=cv2.INTER_NEAREST)  # 使用resize()函数可替代，interpolation选择插值的方式
print('原图尺寸：', img.shape)
print(empty_img.shape)
cv2.imshow('img', img)
cv2.imshow('empty_img', empty_img)
cv2.waitKey(0)  # 在程序中添加一个等待键盘输入的循环，等待用户按下任意键后关闭图像窗口;不加会瞬间自动关闭打开的图像
