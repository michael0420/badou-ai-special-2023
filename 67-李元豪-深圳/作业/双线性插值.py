# -*- coding: utf-8 -*-  
__author__ = '李元豪 from https://www.zhilu.space'

import numpy as np  # 引入numpy库，用于处理数值计算
import math  # 引入math库，用于处理数学计算
import cv2  # 引入opencv库，用于处理图像


def double_linear(input_signal , zoom_multiples) :
    '''
    双线性插值
    :param input_signal: 输入图像
    :param zoom_multiples: 放大倍数
    :return: 双线性插值后的图像
    '''
    input_signal_cp = np.copy (input_signal)  # 输入图像的副本

    input_row , input_col = input_signal_cp.shape  # 获取输入图像的尺寸（行、列）

    # 输出图像的尺寸
    output_row = int (input_row * zoom_multiples)
    output_col = int (input_col * zoom_multiples)

    output_signal = np.zeros ((output_row , output_col))  # 初始化输出图像

    # 对输出图像中的每一个像素进行双线性插值
    for i in range (output_row) :
        for j in range (output_col) :
            # 输出图片中坐标 （i，j）对应至输入图片中的最近的四个点（x1，y1）（x2, y2），（x3， y3），(x4，y4)的均值
            temp_x = i / output_row * input_row
            temp_y = j / output_col * input_col

            x1 = int (temp_x)
            y1 = int (temp_y)

            x2 = x1
            y2 = y1 + 1

            x3 = x1 + 1
            y3 = y1

            x4 = x1 + 1
            y4 = y1 + 1

            u = temp_x - x1
            v = temp_y - y1

            # 当x4或y4超过输入图像的大小时，需要防止越界，这时需要调整x1,x2,x3,x4,y1,y2,y3,y4的值以保证它们都在输入图像内。
            if x4 >= input_row :
                x4 = input_row - 1
                x2 = x4
                x1 = x4 - 1
                x3 = x4 - 1
            if y4 >= input_col :
                y4 = input_col - 1
                y3 = y4
                y1 = y4 - 1
                y2 = y4 - 1

                # 利用双线性插值计算输出图像中（i，j）点的像素值，该插值考虑了四个邻近点的像素值。
            output_signal[i , j] = (1 - u) * (1 - v) * int (input_signal_cp[x1 , y1]) + (1 - u) * v * int (
                input_signal_cp[x2 , y2]) + u * (1 - v) * int (input_signal_cp[x3 , y3]) + u * v * int (input_signal_cp[
                                                                                                            x4 , y4])  # (1-u)*(1-v)表示左上角的权重，u*v表示右下角的权重。其他类似。   根据四个邻近点的像素值和对应的权重进行插值计算。
    return output_signal  # 返回插值后的图像。 输出图像的每个像素值都是通过双线性插值算法从输入图像的四个邻近像素计算得出的。这个方法的结果比简单复制输入图像或使用最近邻插值的结果更平滑和自然。
# Read image
img = cv2.imread(r"D:\AI1026\a22.png",0).astype(np.cfloat)
out = double_linear(img,2).astype(np.uint8)
# Save result
cv2.imshow("result", out)
cv2.imwrite("out.jpg", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

