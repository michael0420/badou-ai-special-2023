import cv2
import numpy as np


def nearest_interpolation(img, a, b):
    height, width, channels = img.shape
    # np.uint8是numpy库中的一个数据类型，代表一个长度为8位的无符号整数。它可以存储的数据范围是0到255之间的整数，共256个不同的值
    e_image = np.zeros((a, b, channels), np.uint8)
    # 等比例缩放比
    sh = height / a
    sw = width / b
    for c in range(channels):
        for i in range(a):
            for j in range(b):
                # e_image在img坐标系中的坐标
                x = int(i * sh + 0.5)  # int(),四舍五入转为整型，使用向下取整。
                y = int(j * sw + 0.5)
                e_image[i, j, c] = img[x, y, c]
    return e_image


if __name__ == '__main__':
    img = cv2.imread('D:/cv_project/demo_1/lenna.png')
    e_img = nearest_interpolation(img, 200, 200)
    cv2.imshow('beauty', e_img)
    cv2.imshow('src_beauty', img)
    cv2.waitKey()
