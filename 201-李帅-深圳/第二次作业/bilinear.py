# 实现双线性插值算法

import cv2
import numpy as np


def bilinear(input_img, out_dim):
    src_h, src_w, channel = input_img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]

    if src_h == dst_h and src_w == dst_w:
        return input_img.copy()

    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h

    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 计算新图像对应的原始图像点
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 计算新像素点周围四个点坐标
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 计算新像素点加权后的像素图像
                temp0 = (src_x1 - src_x) * input_img[src_y0, src_x0, i] + (src_x - src_x0) * input_img[
                    src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * input_img[src_y1, src_x0, i] + (src_x - src_x0) * input_img[
                    src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('tiger.png')
    dst = bilinear(img, (1000, 1000))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()
