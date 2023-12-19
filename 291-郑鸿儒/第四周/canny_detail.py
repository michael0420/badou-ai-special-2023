#!/usr/bin/env python
# encoding=utf-8
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


if "__main__" == __name__:
    # img_gray = cv2.imread("img/lenna.png", 0)
    # img = plt.imread("img/lenna.png")
    # gray = cv2.cvtColor(img * 255, cv2.COLOR_RGB2GRAY)

    pic_path = 'img/lenna.png'
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    gray = img.mean(axis=-1)  # 取均值就是灰度化了
    # print(img)

    # cv2.imshow('gray', img_gray)
    # cv2.waitKey(0)
    # print(gray * 255)
    sigma = 0.5  # 经验值
    dim = round(sigma * 6) + 1  # 经验做法sigma值乘一个系数加一个常数
    # 非奇数时应变为奇数
    if not dim % 2:
        dim += 1
    # 定义高斯卷积核
    Gaussian_filter = np.zeros((dim, dim))
    # 定义序列，中心为0，用于生成高斯函数值
    dim_arr = [i - dim // 2 for i in range(dim)]
    # G(x, y) = 1 / 2 * pi * sigma * exp(-(x ** 2 + y ** 2) / 2 * sigma ** 2)
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    # 计算高斯卷积核
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp((dim_arr[i] ** 2 + dim_arr[j] ** 2) * n2)
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    # pad dim // 2 层0
    img_pad = np.pad(gray, dim // 2, "constant")
    # print(img_pad)
    # 计算高斯平滑
    img_new = np.zeros(gray.shape)
    print(Gaussian_filter)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            img_new[i, j] = np.sum(img_pad[i: i + dim, j: j + dim] * Gaussian_filter)
    print(img_new)
    # sobel算子
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # 计算梯度
    img_grade_x = np.zeros(img_new.shape)
    img_grade_y = np.zeros(img_new.shape)
    img_grade_diag = np.zeros(img_new.shape)
    img_new_pad = np.pad(img_new, 1, "constant")
    # print(img_new)
    for i in range(img_new.shape[0]):
        for j in range(img_new.shape[1]):
            img_grade_x[i, j] = np.sum(img_new_pad[i: i + 3, j: j + 3] * sobel_kernel_x)
            img_grade_y[i, j] = np.sum(img_new_pad[i: i + 3, j: j + 3] * sobel_kernel_y)
            img_grade_diag[i, j] = np.sqrt(img_grade_x[i, j] ** 2 + img_grade_y[i, j] ** 2)
    img_grade_x[img_grade_x == 0] = 0.00000001
    angle = img_grade_y / img_grade_x
    # print(img_grade_diag)

    # 非极大值抑制
    img_supression = np.zeros(img_new.shape)
    for i in range(1, img_new.shape[0] - 1):
        for j in range(1, img_new.shape[1] - 1):
            temp = img_grade_diag[i - 1: i + 2, j - 1: j + 2]
            # print(1, i, j)
            flag = True
            if angle[i, j] <= -1:
                # 梯度与坐标轴的夹角（锐角）大小为tan(α - 90°),故有(0, 0) 对应权重是 - 1 / tanα
                # (0, 1) 对应权重是1 + 1 / tanα
                grade_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                grade_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_grade_diag[i, j] > grade_1 and img_grade_diag[i, j] > grade_2):
                    flag = False
            elif angle[i, j] >= 1:
                grade_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                grade_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_grade_diag[i, j] > grade_1 and img_grade_diag[i, j] > grade_2):
                    flag = False
            elif angle[i, j] > 0:
                grade_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                grade_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                # print(img_grade_diag[i, j])
                # print(grade_1, grade_2)
                if not (img_grade_diag[i, j] > grade_1 and img_grade_diag[i, j] > grade_2):
                    flag = False
            elif angle[i, j] < 0:
                grade_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                grade_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_grade_diag[i, j] > grade_1 and img_grade_diag[i, j] > grade_2):
                    flag = False
            if flag:
                img_supression[i, j] = img_grade_diag[i, j]

    # 双阈值检测
    lower_boundary = img_grade_diag.mean() * 0.5
    higher_boundary = lower_boundary * 3
    print(lower_boundary, higher_boundary)
    stack = []
    for i in range(1, img_supression.shape[0] - 1):
        for j in range(1, img_supression.shape[1] - 1):
            if img_supression[i, j] > higher_boundary:
                img_supression[i, j] = 255
                stack.append([i, j])
            elif img_supression[i, j] < lower_boundary:
                img_supression[i, j] = 0

    while not len(stack) == 0:
        tmp_x, tmp_y = stack.pop()
        for i in range(tmp_x - 1, tmp_x + 2):
            for j in range(tmp_y - 1, tmp_y + 2):
                if higher_boundary > img_supression[i, j] > lower_boundary:
                    stack.append([i, j])
                    img_supression[i, j] = 255

    for i in range(img_supression.shape[0]):
        for j in range(img_supression.shape[1]):
            if img_supression[i, j] != 0 and img_supression[i, j] != 255:
                img_supression[i, j] = 0

    plt.figure()
    plt.imshow(img_supression.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()