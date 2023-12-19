import math
import numpy as np

import matplotlib.pyplot as plt


def to_graying(pic_path):
    # 这里使用plt读取图片
    org_img = plt.imread(pic_path)
    # 这里如果是png的图片内部的数据是0~1的浮点数，这里要先进行扩展到0~255再进行计算
    if pic_path[-4:] == '.png':
        org_img = org_img * 255
    # 图像灰度化，这里axis为-1或者2来计算RGB的均值
    org_img = org_img.mean(axis=-1)
    return org_img


def Gaussian(sigma):
    # int(round(6×σ+1))高斯滤波的大小公式
    dim = int(np.round(6 * sigma + 1))
    # 这里让高斯滤波后面的数组为奇数方便后续计算
    if dim % 2 == 0:
        dim += 1
    # 生成高斯滤波的卷积
    gaussian_filter = np.zeros([dim, dim])
    # 生成一个高斯滤波相对于中心坐标的一维数组，后续公式中的x,y从这里取
    tmp = [i - dim // 2 for i in range(dim)]
    # 计算常数部分1/2piσ平方
    n1 = 1 / (2 * math.pi * sigma ** 2)
    # 计算系数的指数的分母部分
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            # 这里是计算高斯然后写入数组中
            gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    # 这里让高斯归一化使所有元素之和等于1
    gaussian_filter = gaussian_filter / gaussian_filter.sum()
    return dim, gaussian_filter


def padding(img, dim, gaussian):
    dx, dy = img.shape
    # 创建一个大小相同的空矩阵
    new_img = np.zeros(img.shape)
    # 计算高斯滤波的半径
    radius = dim // 2
    img_pad = np.pad(img, ((radius, radius), (radius, radius)), 'constant')
    for i in range(dx):
        for j in range(dy):
            # 卷积*高斯滤波然后求和
            new_img[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * gaussian)
    plt.figure(1)
    plt.imshow(new_img.astype(np.uint8), cmap='gray')
    # plt.axis('off')
    # plt.show()
    return new_img


def gradient(img, img_padding):
    dx, dy = img.shape
    # sobel算子
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # 存储梯度值
    img_gradient_x = np.zeros(img_padding.shape)
    img_gradient_y = np.zeros([dx, dy])
    # 存储梯度幅值
    img_gradient = np.zeros(img_padding.shape)
    # 因为我们之前高斯图像是让其填充为整数，这里的sobel为奇数所以需要再填充一行一列
    img_gradient_padding = np.pad(img_padding, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            # x方向
            img_gradient_x[i, j] = np.sum(img_gradient_padding[i:i + 3, j:j + 3] * sobel_kernel_x)
            # y方向
            img_gradient_y[i, j] = np.sum(img_gradient_padding[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_gradient[i, j] = np.sqrt(img_gradient_x[i, j] ** 2 + img_gradient_y[i, j] ** 2)
    # 这里是为了防止出现为0的情况，这里用极小值代替
    img_gradient_x[img_gradient_x == 0] = 0.00000001
    # 计算梯度的方向
    angle = img_gradient_y / img_gradient_x
    plt.figure(2)
    plt.imshow(img_gradient.astype(np.uint8), cmap='gray')
    # plt.axis('off')
    # plt.show()
    return angle, img_gradient


'''
传入梯度图像计算非极大值抑制
'''


def non_maxima(img, angle, img_gradient):
    dx, dy = img.shape
    img_yizhi = np.zeros(img_gradient.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            # 用于标记在8邻域中是否进行抹除
            flag = True
            # 获取梯度增幅的八邻域矩阵,这里因为是从1开始的，所以可以减1
            # 这里的矩阵大小与sobel一致
            tmp = img_gradient[i - 1:i + 2, j - 1:j + 2]
            # 使用线性插值判断是否抑制
            if angle[i, j] <= -1:
                num_1 = (tmp[0, 1] - tmp[0, 0] / angle[i, j] + tmp[0, 1])
                num_2 = (tmp[2, 1] - tmp[2, 2] / angle[i, j] + tmp[2, 1])
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] < num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (tmp[0, 2] - tmp[0, 1]) / angle[i, j] + tmp[0, 1]
                num_2 = (tmp[2, 0] - tmp[2, 1]) / angle[i, j] + tmp[2, 1]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (tmp[0, 2] - tmp[1, 2]) * angle[i, j] + tmp[1, 2]
                num_2 = (tmp[2, 0] - tmp[1, 0]) * angle[i, j] + tmp[1, 0]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (tmp[1, 0] - tmp[0, 0]) * angle[i, j] + tmp[1, 0]
                num_2 = (tmp[1, 2] - tmp[2, 2]) * angle[i, j] + tmp[1, 2]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_gradient[i, j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    # plt.axis('off')
    return img_yizhi


'''
双阈值检测
'''


def dual_threshold(img_gradient, img_yizhi):
    lower_boundary = img_gradient.mean() * 0.5
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
    # 绘图
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    # plt.axis('off')


if __name__ == "__main__":
    # 读取照片
    picture_path = "../../lenna.png"
    # 图片灰度化
    img_gray = to_graying(picture_path)
    # 得到高斯滤波
    dim_value, gaussian_filter = Gaussian(0.5)
    # 用高斯滤波对图片进行平滑处理
    img_gaussian_padding = padding(img_gray, dim_value, gaussian_filter)
    # 梯度化图像
    gradient_angle, gradient_img = gradient(img_gray, img_gaussian_padding)
    # 非极大值抑制
    img_yizhi_value = non_maxima(img_gray, gradient_angle, gradient_img)
    # 双阈值检测
    dual_threshold(gradient_img, img_yizhi_value)
    plt.show()
