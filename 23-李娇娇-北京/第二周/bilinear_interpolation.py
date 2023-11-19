# -*- coding: utf-8 -*-

# @FileName: bilinear_interpolation.py
# @Time    : 2023/10/31 5:13 PM
# @Author  : lijiaojiao
# @Software: PyCharm
import cv2
import numpy as np


def bilinear_interpolation(img, dst_img):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = dst_img.shape[:2]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 几何中心重合找出目标图片坐标点对应的源图像坐标点
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 找出将用于计算插值的点的坐标
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 计算插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst_img = np.zeros((800, 800, 3), dtype=np.uint8)
    dst_img = bilinear_interpolation(img, dst_img)
    cv2.imshow('img', img)
    cv2.imshow('bilinear interp',dst_img)
    cv2.waitKey()

