import math
import cv2
import matplotlib.pyplot as plt

import numpy as np

img_org = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
print(img_gray)
# 1. 构造高斯平滑滤波核
# 核大小 n*n
sigma = 0.5
n = int(round(6 * sigma + 1))
if n % 2 == 0:
    n += 1

gaussian_filter = np.zeros([n, n])
tmp = [i - n // 2 for i in range(n)]  # 生成一个序列
n1 = 1 / (2 * math.pi * sigma ** 2)
n2 = -1 / (2 * sigma ** 2)

for i in range(n):
    for j in range(n):
        gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))

gaussian_filter = gaussian_filter / gaussian_filter.sum()

print("gaussian_filter\n", gaussian_filter)

dx, dy = img_gray.shape
tmp = n // 2
img_new = np.zeros(img_gray.shape)
img_gray_padding = np.pad(img_gray, pad_width=((tmp, tmp), (tmp, tmp)), mode="constant")  # pad_width 代表上下左右填充padding个数

for i in range(dx):
    for j in range(dy):
        img_new[i, j] = int(np.sum(
            img_gray_padding[i:i + n, j:j + n] * gaussian_filter))  # 对每个像素进行n*n step = 1 的卷积求和，赋值给新像素点

# cv2.imshow("ln", img_new.astype(np.uint8))
# print(img_new)
# cv2.waitKey(0)
# soble计算边缘
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
img_tidu_x = np.zeros(img_new.shape)
img_tidu_y = np.zeros(img_new.shape)
img_tidu = np.zeros(img_new.shape)

img_new_padding = np.pad(img_new, ((3, 3), (3, 3)), "constant")
for i in range(img_new.shape[0]):
    for j in range(img_new.shape[1]):
        img_tidu_x[i, j] = np.sum(img_new_padding[i:i + 3, j:j + 3] * sobel_kernel_x)
        img_tidu_y[i, j] = np.sum(img_new_padding[i:i + 3, j:j + 3] * sobel_kernel_y)
        img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)

img_tidu_x[img_tidu_x == 0] = 0.0000001

angle = img_tidu_y / img_tidu_x


# 非极大值抑制
img_yizhi = np.zeros(img_tidu.shape)
dx, dy = img_tidu.shape
for i in range(1, dx - 1):
    for j in range(1, dy - 1):
        flag = True  # 在8邻域内是否要抹去做个标记
        temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
        if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] >= 1:
            num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] > 0:
            num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] < 0:
            num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        if flag:
            img_yizhi[i, j] = img_tidu[i, j]
# cv2.imshow("yizhi", img_yizhi)
# cv2.waitKey(0)

# 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
lower_boundary = img_tidu.mean() * 0.5
high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍

zhan = []
# 首先把高于高阈值的都标记为255，同时进栈
for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
    for j in range(1, img_yizhi.shape[1] - 1):
        if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
            img_yizhi[i, j] = 255
            zhan.append([i, j])
        elif img_yizhi[i, j] <= lower_boundary:  # 舍
            img_yizhi[i, j] = 0


cv2.imshow("result", img_yizhi)
cv2.waitKey(0)

# 对栈内元素 和 周围的8个相邻点进行比较，如果当前点的周围有比它大的值，就认为是强点，否则可能是噪音，需要抑制
while not len(zhan) == 0:
    temp_1, temp_2 = zhan.pop()  # 出栈
    a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
        zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
    if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2] = 255
        zhan.append([temp_1 - 1, temp_2])
    if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 + 1] = 255
        zhan.append([temp_1 - 1, temp_2 + 1])
    if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
        img_yizhi[temp_1, temp_2 - 1] = 255
        zhan.append([temp_1, temp_2 - 1])
    if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
        img_yizhi[temp_1, temp_2 + 1] = 255
        zhan.append([temp_1, temp_2 + 1])
    if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 - 1] = 255
        zhan.append([temp_1 + 1, temp_2 - 1])
    if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2] = 255
        zhan.append([temp_1 + 1, temp_2])
    if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 + 1] = 255
        zhan.append([temp_1 + 1, temp_2 + 1])

for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
            img_yizhi[i, j] = 0


cv2.imshow("result", img_yizhi)
cv2.waitKey(0)