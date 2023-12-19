import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from numpy import ndarray

if __name__ == '__main__':
    ImgPath = 'F:/LCE_Test/AI_Homework/AI_Homework/HomeworkImg.jpg'
    RgbImg = cv2.imread(ImgPath)
    GrayImg = cv2.cvtColor(RgbImg, cv2.COLOR_BGR2GRAY)

    # 高斯滤波
    Sigma = 0.5
    dim = int(np.round(6 * Sigma + 1))
    if dim % 2 == 0:
        dim = dim + 1
    Gaussian_filter = np.zeros([dim, dim])
    temp = [i - dim // 2 for i in range(dim)]
    n1 = 1 / (2 * math.pi * Sigma ** 2)
    n2 = -1 / (2 * Sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (temp[i] ** 2 + temp[j] ** 2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    dx, dy = GrayImg.shape
    Img_New = np.zeros(GrayImg.shape)
    temp = dim // 2
    img_pad = np.pad(GrayImg, ((temp, temp), (temp, temp)), 'constant')  # 填充
    for i in range(dx):
        for j in range(dy):
            Img_New[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
    plt.figure(1)
    plt.imshow(Img_New.astype(np.uint8), cmap='gray')

    # 求梯度
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Img_gradient_x = np.zeros([dx, dy])
    Img_gradient_y = np.zeros([dx, dy])
    Img_gradient = np.zeros([dx, dy])
    img_pad = np.pad(Img_New, ((1, 1), (1, 1)), 'constant')
    angle = np.zeros([dx, dy])
    for i in range(dx):
        for j in range(dy):
            Img_gradient_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
            Img_gradient_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            Img_gradient[i, j] = np.sqrt(Img_gradient_x[i, j] ** 2 + Img_gradient_y[i, j] ** 2)
            # angle[i, j] = math.degrees(math.atan(Img_gradient_y[i, j]/Img_gradient_x[i, j]))
    Img_gradient_x[Img_gradient_x == 0] = 0.00000001
    Angle = Img_gradient_y / Img_gradient_x  # arctan（Gy/Gx)
    plt.figure(2)
    plt.imshow(Img_gradient.astype(np.uint8), cmap='gray')

    # 非极大值抑制
    Img_Suppression = np.zeros([dx, dy])
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True
            temp = Img_gradient[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值8邻域矩阵
            if Angle[i, j] <= -1:
                num1 = (temp[0, 1] - temp[0, 0]) / Angle[i, j] + temp[0, 1]
                num2 = (temp[2, 1] - temp[2, 2]) / Angle[i, j] + temp[2, 1]
                if not (Img_gradient[i, j] > num1 and Img_gradient[i, j] > num2):
                    flag = False
            elif Angle[i, j] >= 1:
                num1 = (temp[0, 2] - temp[0, 1]) / Angle[i, j] + temp[0, 1]
                num2 = (temp[2, 0] - temp[2, 1]) / Angle[i, j] + temp[2, 1]
                if not (Img_gradient[i, j] > num1 and Img_gradient[i, j] > num2):
                    flag = False
            elif Angle[i, j] > 0:
                num1 = (temp[0, 2] - temp[1, 2]) * Angle[i, j] + temp[1, 2]
                num2 = (temp[2, 0] - temp[1, 0]) * Angle[i, j] + temp[1, 0]
                if not (Img_gradient[i, j] > num1 and Img_gradient[i, j] > num2):
                    flag = False
            elif Angle[i, j] < 0:
                num1 = (temp[1, 0] - temp[0, 0]) * Angle[i, j] + temp[1, 0]
                num2 = (temp[1, 2] - temp[2, 2]) * Angle[i, j] + temp[1, 2]
                if not (Img_gradient[i, j] > num1 and Img_gradient[i, j] > num2):
                    flag = False
            if flag:
                Img_Suppression[i, j] = Img_gradient[i, j]
    plt.figure(3)
    plt.imshow(Img_Suppression.astype(np.uint8), cmap='gray')

    # 双阈值
    lower_boundary = Img_gradient.mean() * 0.5
    higher_boundary = lower_boundary * 3.5
    TempValue = []
    for i in range(1, Img_Suppression.shape[0] - 1):
        for j in range(1, Img_Suppression.shape[1] - 1):
            if Img_Suppression[i, j] >= higher_boundary:
                Img_Suppression[i, j] = 255
                TempValue.append([i, j])
            elif Img_Suppression[i, j] <= lower_boundary:
                Img_Suppression[i, j] = 0

    # 弱边缘点使用八邻域
    while not len(TempValue) == 0:
        temp1, temp2 = TempValue.pop()  # 移除元素（默认从最后元素开始）
        a = Img_Suppression[temp1 - 1:temp1 + 2, temp2 - 1:temp2 + 2]
        if (a[0, 0] < higher_boundary) and (a[0, 0] > lower_boundary):
            Img_Suppression[temp1 - 1, temp2 - 1] = 255  # 该像素点标记为边缘
            TempValue.append([temp1 - 1, temp2 - 1])     # 末尾增加该坐标
        if (a[0, 1] < higher_boundary) and (a[0, 1] > lower_boundary):
            Img_Suppression[temp1 - 1, temp2] = 255
            TempValue.append([temp1 - 1, temp2])
        if (a[0, 2] < higher_boundary) and (a[0, 2] > lower_boundary):
            Img_Suppression[temp1 - 1, temp2 + 1] = 255
            TempValue.append([temp1 - 1, temp2 + 1])
        if (a[1, 0] < higher_boundary) and (a[1, 0] > lower_boundary):
            Img_Suppression[temp1, temp2 - 1] = 255
            TempValue.append([temp1, temp2 - 1])
        if (a[1, 2] < higher_boundary) and (a[1, 2] > lower_boundary):
            Img_Suppression[temp1, temp2 + 1] = 255
            TempValue.append([temp1, temp2 + 1])
        if (a[2, 0] < higher_boundary) and (a[2, 0] > lower_boundary):
            Img_Suppression[temp1 + 1, temp2 - 1] = 255
            TempValue.append([temp1 + 1, temp2 - 1])
        if (a[2, 1] < higher_boundary) and (a[2, 1] > lower_boundary):
            Img_Suppression[temp1 + 1, temp2] = 255
            TempValue.append([temp1 + 1, temp2])
        if (a[2, 2] < higher_boundary) and (a[2, 2] > lower_boundary):
            Img_Suppression[temp1 + 1, temp2 + 1] = 255
            TempValue.append([temp1 + 1, temp2 + 1])
    for i in range(Img_Suppression.shape[0]):
        for j in range(Img_Suppression.shape[1]):
            if Img_Suppression[i, j] != 0 and Img_Suppression[i, j] != 255:
                Img_Suppression[i, j] = 0
    plt.figure(4)
    plt.imshow(Img_Suppression.astype(np.uint8), cmap='gray')
