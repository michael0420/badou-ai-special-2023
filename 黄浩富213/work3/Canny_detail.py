import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

img = cv2.imread('lenna.png')
# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对图像进行高斯平滑
sigma = 0.5
kernel_size = int(np.round(sigma * 6 + 1))
if kernel_size % 2 == 0:
    kernel_size + 1

center = kernel_size // 2
kernel = np.zeros([kernel_size, kernel_size])
for i in range(kernel_size):
    for j in range(kernel_size):
        x = i - center
        y = j - center
        kernel[i, j] = (1 / 2 * math.pi * sigma ** 2) * math.exp((-(x ** 2) + (y ** 2)) / (2 * sigma ** 2))
kernel =  kernel / kernel.sum()
dx = img.shape[0]
dy = img.shape[1]

img_new = np.zeros(img.shape)
tmp = kernel_size // 2
img_padded = np.pad(gray,((tmp, tmp), (tmp, tmp)), 'constant')
for i in range(dx):
    for j in range(dy):
        img_new[i, j] = np.sum(img_padded[i:i + kernel_size, j:j + kernel_size] * kernel)
img_new.astype(np.uint8)
# 2、求梯度
kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
kernel_y = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
grad_x = np.zeros(img_new.shape)
grad_y = np.zeros([dx, dy])
grad = np.zeros(img_new.shape)
img_padded = np.pad(img_new,((1, 1), (1, 1)), 'constant')
for i in range(dx):
    for j in range(dy):
        grad_x[i, j] = np.sum(img_padded[i:i + 3, j: j + 3] * kernel_x)
        grad_y[i, j] = np.sum(img_padded[i:i + 3, j: j + 3] * kernel_y)
        grad[i, j] = np.sqrt(grad_x[i, j]**2 + grad_y[i, j]**2)

grad_x[grad_x == 0] = 0.00000001
angle = grad_y / grad_x
grad.astype(np.uint8)
# 3、非极大值抑制
img_yizhi = np.zeros(grad.shape)
for i in range(1, dx - 1):
    for j in range(1, dy - 1):
        flag = True  # 在8邻域内是否要抹去做个标记
        temp = grad[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
        if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (grad[i, j] > num_1 and grad[i, j] > num_2):
                flag = False
        elif angle[i, j] >= 1:
            num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if not (grad[i, j] > num_1 and grad[i, j] > num_2):
                flag = False
        elif angle[i, j] > 0:
            num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            if not (grad[i, j] > num_1 and grad[i, j] > num_2):
                flag = False
        elif angle[i, j] < 0:
            num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if not (grad[i, j] > num_1 and grad[i, j] > num_2):
                flag = False
        if flag:
            img_yizhi[i, j] = grad[i, j]
plt.figure(3)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')

# 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
lower_boundary = grad.mean() * 0.5
high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
zhan = []
for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
    for j in range(1, img_yizhi.shape[1] - 1):
        if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
            img_yizhi[i, j] = 255
            zhan.append([i, j])
        elif img_yizhi[i, j] <= lower_boundary:  # 舍
            img_yizhi[i, j] = 0

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
cv2.imshow('gard', grad)
cv2.imshow('img', img_new)
cv2.imshow('Original', img)
cv2.imshow('Gray', gray)

cv2.waitKey(0)