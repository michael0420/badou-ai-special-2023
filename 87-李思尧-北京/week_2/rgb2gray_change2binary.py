import cv2
import numpy as np


def rgb_to_gray(img):
    # 获取图片每个像素, x是行, y是列
    x, y = img.shape[:2]
    # 定义一个空图片，和输入图片大小，数据类型一样
    img_gray = np.zeros([x, y], img.dtype)  # np.zeros的用法，定义一个空矩阵
    # 循环矩阵，即各坐标像素点
    for i in range(x):
        for j in range(y):
            m = img[i, j]  # 取出当前行列坐标中的像素点的BGR值
            # 将BGR坐标转化为gray坐标并赋值给新图像，opencv是BGR非RGB，转化方法是科学经验
            img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
    return img_gray


def change_to_binary(img_gray):
    x, y = img_gray.shape[:2]
    img_binary = np.zeros([x, y])
    for i in range(x):
        for j in range(y):
            if img_gray[i, j] <= 128:
                img_binary[i, j] = 0
            else:
                img_binary[i, j] = 1
    return img_binary


if __name__ == '__main__':
    img = cv2.imread('D:/cv_project/demo_1/lenna.png')
    img_gray = rgb_to_gray(img)
    cv2.imshow('gray beauty', img_gray)
    img_binary = change_to_binary(img_gray)
    cv2.imshow('binary beauty', img_binary)
    cv2.waitKey()

    pass
