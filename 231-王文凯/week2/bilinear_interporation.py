import cv2
import numpy as np


def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim
    if src_h == dst_h and src_w == dst_w:
        return img
    dst_img = np.zeros((dst_h, dst_w, channel), img.dtype)
    scale_x, scale_y = src_w / dst_w, src_h / dst_h
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 几何中心重合
                src_x, src_y = (dst_x + 0.5) * scale_x - 0.5, (dst_y + 0.5) * scale_y - 0.5
                # 向下取整 四点坐标初始化
                src_x0, src_y0 = int(np.floor(src_x)), int(np.floor(src_y))
                src_x1, src_y1 = min(src_x0 + 1, src_w - 1), min(src_y0 + 1, src_h - 1)
                # tmp1 f(R1) ((x1 - x) / (x1 - x0)) * f(Q11) + ((x - x0) / (x1 - x0)) * f(Q21)
                # tmp2 f(R2) ((x1 - x) / (x1 - x0)) * f(Q12) + ((x - x0) / (x1 - x0)) * f(Q22)
                # x1 - x0 = 1 省略
                tmp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                tmp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                # f(x, y) = ((y1 - y) / (y1 - y0)) * f(R1) + ((y - y0) / (y1 - y0)) * f(R2)
                # 同理 y1 - y0 = 1 省略
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * tmp0 + (src_y - src_y0) * tmp1)
    return dst_img


if __name__ == "__main__":
    lenna = cv2.imread('../Images/lenna.png')
    lenna_change = bilinear_interpolation(lenna, (300, 300))
    cv2.imshow('lenna_change', lenna_change)
    cv2.waitKey()
