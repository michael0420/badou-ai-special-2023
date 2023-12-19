import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

def showImg(num,img):
    plt.figure(num)
    plt.imshow(img.astype(np.uint8),cmap="gray")


def get_gaussian_filter(sigma,kernal_size):
    # 存储高斯核
    gaussian_filter = np.zeros([kernel_size, kernel_size])
    # 计算要padding的值 ，// 为向下取证
    pad = kernal_size // 2
    n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核
    n2 = -1 / (2 * sigma ** 2)
    for i in range(-pad, -pad+kernal_size):
        for j in range(-pad, -pad + kernal_size):
            gaussian_filter[i,j] = n1 * math.exp(n2 * (i ** 2 + j ** 2))
    gaussian_filter = gaussian_filter / gaussian_filter.sum()
    #print("高斯滤波",gaussian_filter)
    return gaussian_filter


def get_gaussian_img(img, gaussian_filter):
    #先做0填充
    H,W = img.shape[:2]
    pad = gaussian_filter.shape[0] // 2
    pad_img = np.zeros((H+2*pad,W+2*pad))
    pad_img[pad:pad+H, pad:pad+W] = img.copy()
    gaussian_img = np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    # 做卷积
    for i in range(H):
        for j in range(W):
            gaussian_img[i, j] = np.sum(pad_img[i:i + kernel_size, j:j + kernel_size] * gaussian_filter)
    showImg(2,gaussian_img)
    return gaussian_img


def get_sobel_img(gaussian_img):
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    H,W = gaussian_img.shape[:2]
    img_tidu_x = np.zeros([H, W])
    img_tidu_y = np.zeros([H, W])
    img_tidu = np.zeros([H, W])
    pad = sobel_kernel_x.shape[0]
    pad_img = np.zeros((H + 2 * pad, W + 2 * pad))
    pad_img[pad:pad + H, pad:pad + W] = gaussian_img.copy()
    for i in range(H):
        for j in range(W):
            img_tidu_x[i, j] = np.sum(pad_img[i:i + 3, j:j + 3] * sobel_kernel_x)
            img_tidu_y[i, j] = np.sum(pad_img[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
    # 除数不能为0
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y / img_tidu_x
    showImg(3,img_tidu)
    return angle,img_tidu


def get_yizhi_img():
    img_yizhi = np.zeros(img_tidu.shape)
    H,W = img_tidu.shape[:2]
    for i in range(1, H- 1):
        for j in range(1, W - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            # 说明 该梯度值在 第二象限
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            # 说明 该梯度值在 第一象限
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
    showImg(4,img_yizhi)
    return img_yizhi


def get_yuzhi_img(img_yizhi):
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3
    zhan = []
    for i in range(img_yizhi.shape[0] ):
        for j in range( img_yizhi.shape[1]):
            if img_yizhi[i, j] >= high_boundary:
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
    # 7、其他非边缘的点置为0
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
    showImg(5,img_yizhi)

    return img_yizhi


def set_zero(yizhi_img):
    pass


if __name__ == '__main__':
    #1、获取图片灰度化
    img = cv2.imread("lenna.png")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    showImg(1,gray)
    #2、获取高斯滤波
    # round是四舍五入函数，根据标准差求高斯核是几乘几的，也就是维度
    sigma = 0.5
    kernel_size = int(np.round(6 * sigma + 1))
    if kernel_size % 2 == 0:  # 最好是奇数,不是的话加一
        kernel_size += 1
    gaussian_filter = get_gaussian_filter(sigma,kernel_size)
    # 3、用高斯滤波对灰度图片做卷积
    gaussian_img = get_gaussian_img(gray, gaussian_filter)

    # 4、获取每个像素的梯度
    angle,img_tidu = get_sobel_img(gaussian_img)

    # 5、用非极大值抑制
    yizhi_img = get_yizhi_img()

    # 6、双阈值检测遍历所有一定是边的点,查看8邻域是否存在有可能是边的点
    get_yuzhi_img(yizhi_img)

    #最后plt 一并显示

    plt.show()




