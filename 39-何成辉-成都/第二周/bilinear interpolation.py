#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

"""
@author: BraHitYQ
双线性插值（python implementation of bilinear interpolation）
"""


def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape  # 获取输入图像的高度、宽度和通道数
    dst_h, dst_w = out_dim[1], out_dim[0]  # 获取输出图像的高度和宽度
    print("src_h, src_w = ", src_h, src_w)  # 打印输入图像的高度和宽度
    print("dst_h, dst_w = ", dst_h, dst_w)  # 打印输出图像的高度和宽度
    if src_h == dst_h and src_w == dst_w:
        return img.copy()  # 如果输入图像和输出图像尺寸相同，则直接返回输入图像的副本
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)  # 创建一个与输出图像尺寸相同的全零数组作为目标图像
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h  # 计算缩放比例
    for i in range(3):  # 对于每个通道进行插值
        for dst_y in range(dst_h):  # 遍历输出图像的每一行
            for dst_x in range(dst_w):  # 遍历输出图像的每一列
                # 使用几何中心对称法找到对应的源图像坐标
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 找到用于计算插值的源图像坐标点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 计算插值结果
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img  # 返回插值后的图像



if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()
