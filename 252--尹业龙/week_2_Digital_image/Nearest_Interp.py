import cv2
import numpy as np

def nearest_interp(img,out_dim):
    """
    <2023.9.6> version:1.0
    实现最近邻插值算法
    :param:img,输入图片
    :param:out_dim,目标图片维度大小
    :return:empty_img,经插值处理后的图片
    """
    height = out_dim[0]
    width = out_dim[1]
    h, w, channel = img.shape  # 获取输入图片的参数
    empty_img = np.zeros((height, width, channel), np.uint8)  # 新建一个变量用于存储插值后的图片,OpenCV画图中，只支持uint8的数据类型。
    sh = height / h  # height = h * sh,h为输入图像高度，sh为缩放比例
    sw = width / w  # 相当于确定 缩放比例 = 目标图像尺寸 / 原图像尺寸
    for i in range(height):
        for j in range(width):
            x = int(i/sh + 0.5)  # i/sh、j/sw是为了确认要插入的像素点的位置；（判断距离哪个像素点最近）
            y = int(j/sw + 0.5)  # +0.5是为了手动实现向上取整，
            # eg：假设相邻两个像素点位置一个是3，一个是4，那么当计算出的值是3.2时（<0.5）经过+0.5并取整得到的就是3；如果是3.6（>0.5）取整后就是4
            empty_img[i, j] = img[x, y]
    return empty_img
