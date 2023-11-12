import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

# 边缘检测
# step_1 图像灰度化
# step_2 高斯滤波处理图像噪声，平滑图像
# step_3 sobel算子检测图像边缘
# step_4 对梯度幅值进行非极大值抑制
# step_5 双阈值算法检测边缘 连接边缘


def canny_detail(img):
    # step_1 图像灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # step_2 高斯滤波，平滑图像
    # sigma 标准差 用于控制图像的平滑程度 可调
    sigma = 0.5
    # dim 高斯核维度 3sigma原则 根据标准差确定高斯核维度
    dim = int(np.round(6 * sigma + 1))
    # 高斯核维度为偶数时+1
    if dim % 2 == 0:
        dim += 1
    # 高斯核 初始化为全零数组
    gaussian_filter = np.zeros((dim, dim))
    # 临时序列 以每个点为中心坐标计算临域点在高斯函数中的值
    # 本代码中 卷积核为 5 * 5， 中心坐标为[0, 0]，则左上角坐标为[-2, -2]
    # tmp = [-2, -1, 0, 1, 2]
    tmp = [i - dim >> 1 for i in range(dim)]
    # 计算高斯核
    # 公式 G(x,y) = 1 / 2 * Π * σ^2 * e^(-(x^2 + y^2) / (2 * σ^2))
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    # 对高斯核进行归一化
    gaussian_filter /= gaussian_filter.sum()
    # 图像维度
    dx, dy = img_gray.shape
    # img_new 用于存储高斯滤波平滑后图像 初始化
    img_new = np.zeros(img_gray.shape)
    # tmp 临时变量 高斯核中心索引 在进行高斯卷积时，从填充图像中提取局部区域时的偏移量
    tmp = dim >> 1
    # tmp同时也是卷积的零填充时，零的填充个数
    # 边缘填充
    img_pad = np.pad(img_gray, ((tmp, tmp), (tmp, tmp)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * gaussian_filter)

    # step_3 梯度计算 sobel算子进行边缘检测
    # sobel_kernel_x 水平卷积核
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # sobel_kernel_y 垂直卷积核
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # img_gradient_x x方向的梯度图像
    img_gradient_x = np.zeros(img_new.shape)
    # img_gradient_y y方向的梯度图像
    img_gradient_y = np.zeros(img_new.shape)
    # img_gradient 梯度强度图像
    img_gradient = np.zeros(img_new.shape)
    # 边缘填充 由于sobel算子的卷积核确认为 3 * 3， 因此边缘填充 0的填充个数也已经确定 3 >> 1 = 1
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), "constant")
    for i in range(dx):
        for j in range(dy):
            img_gradient_x[i, j] = np.sum(img_pad[i: i + 3, j: j + 3] * sobel_kernel_x)
            img_gradient_y[i, j] = np.sum(img_pad[i: i + 3, j: j + 3] * sobel_kernel_y)
            img_gradient[i, j] = np.sqrt(img_gradient_y[i, j] ** 2 + img_gradient_y[i, j] ** 2)
    # 避免后续计算中的除零错误 将梯度值为0的元素设置为一个很小的值
    img_gradient_x[img_gradient_x == 0] = 0.000000001
    # angle 每个像素位置的梯度方向 正切角度
    angle = img_gradient_y / img_gradient_x

    # step_4 非极大值抑制
    # img_non_maximum_suppression 用于存储非极大值抑制后图像
    img_non_maximum_suppression = np.zeros(img_gradient.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            # flag 标记 用于确认是否要抹去当前像素
            flag = True
            # temp 当前像素在梯度图像中的八邻域矩阵， 用于后续线性插值
            temp = img_gradient[i - 1: i + 2, j - 1: j + 2]
            # 90° < angle < 135°
            if angle[i, j] <= -1:
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 0]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            # 45° < angle < 90°
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            # 0° < angle < 45°
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            # 135° < angle < 180°
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            if flag:
                img_non_maximum_suppression[i, j] = img_gradient[i, j]

    # step_5 双阈值检测
    # lower_boundary 低阈值
    lower_boundary = img_gradient.mean() * 0.5
    # high_boundary 高阈值
    high_boundary = lower_boundary * 3
    # 栈 储存强边缘的点
    stack = []
    # 大于高阈值 强边缘 255 小于高阈值 非边缘 0 中间 弱边缘
    for i in range(1, img_gradient.shape[0] - 1):
        for j in range(1, img_gradient.shape[1] - 1):
            if img_gradient[i, j] >= high_boundary:
                img_non_maximum_suppression[i, j] = 255
                stack.append([i, j])
            elif img_gradient[i, j] <= lower_boundary:
                img_non_maximum_suppression[i, j] = 0

    while len(stack):
        temp_1, temp_2 = stack.pop()
        # 八邻域矩阵
        a = img_non_maximum_suppression[temp_1 - 1: temp_1 + 2, temp_2 - 1: temp_2 + 2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_non_maximum_suppression[temp_1 - 1, temp_2 - 1] = 255
            stack.append([temp_1 - 1, temp_2 - 1])
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_non_maximum_suppression[temp_1 - 1, temp_2] = 255
            stack.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_non_maximum_suppression[temp_1 - 1, temp_2 + 1] = 255
            stack.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_non_maximum_suppression[temp_1, temp_2 - 1] = 255
            stack.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_non_maximum_suppression[temp_1, temp_2 + 1] = 255
            stack.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_non_maximum_suppression[temp_1 + 1, temp_2 - 1] = 255
            stack.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_non_maximum_suppression[temp_1 + 1, temp_2] = 255
            stack.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_non_maximum_suppression[temp_1 + 1, temp_2 + 1] = 255
            stack.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_non_maximum_suppression.shape[0]):
        for j in range(img_non_maximum_suppression.shape[1]):
            if img_non_maximum_suppression[i, j] != 0 and img_non_maximum_suppression[i, j] != 255:
                img_non_maximum_suppression[i, j] = 0

    return img_non_maximum_suppression


img = cv2.imread("../Images/lenna.png")
img_suppression = canny_detail(img)
# img_suppression = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 200, 300)
plt.figure("img_gray")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray')
plt.axis('off')
plt.figure("img_suppression")
plt.imshow(img_suppression.astype(np.uint8), cmap='gray')
plt.axis('off')
plt.show()
