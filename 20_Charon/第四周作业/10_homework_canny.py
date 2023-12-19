import cv2
import numpy as np


# canny计算边缘步骤
# 高斯卷积（1、计算高斯核，2、padding后图像与高斯核进行卷积）
# 计算梯度值和梯度方向 （计算X和Y方向上的梯度值 然后合并得到梯度值并计算梯度方向即tan值）
# 非极大值抑制 （用线性插值方法计算num1num2的值，并和合并后的梯度值比较，取较大的）
# 双阈值和边缘连接 （小于最小阈值的为0 大于最大阈值的给255， 中间态的另讨论）


# 计算高斯核
def gauss_filter(sigma, kernel_size):
    '''
    :param sigma: 标准差
    :param kernel_size: 高斯卷积核的尺寸
    :return: 高斯卷积核kernel
    '''
    center = kernel_size // 2
    kernel = np.zeros([kernel_size, kernel_size])
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))  # 固有公式，如果不记得，直接搜索‘二维高斯函数’
            kernel_sum = np.sum(kernel)
    # 将核归一化
    kernel /= kernel_sum
    return kernel


# 填充图像并高斯卷积
def pad_blur(img, kernel):
    '''
    :param img: 待卷积的图
    :param kernel: 卷积核
    :return: 卷积后的图new_img
    '''
    # 计算padding圈数
    img_h, img_w = img.shape[:2]
    kernel_h, kernel_w = kernel.shape[:2]
    padding = (kernel_size - 1) // 2  # 整除运算
    # 创建最终输出图
    new_img = np.zeros([img_h, img_w])
    # 扩展图像
    pad_img = np.zeros((img_h + padding * 2, img_w + padding * 2))
    pad_img[padding:img_h + padding, padding:img_w + padding] = img
    # 与核进行卷积
    for i in range(img_h):
        for j in range(img_w):
            new_img[i, j] = np.sum(kernel * pad_img[i:i + kernel_size, j:j + kernel_size])
    return new_img


src_img = cv2.imread('lenna2.jpg', 0)
sigma = 5
kernel_size = 3

# 1、计算高斯核
kernel = gauss_filter(sigma, kernel_size)
# 2、padding后图像与高斯核进行卷积
gauss_img = pad_blur(src_img, kernel)

# 3、计算梯度值和梯度方向
# 计算水平和垂直方向上的梯度值
tidu_value_x = cv2.Sobel(gauss_img, cv2.CV_64F, 1, 0, ksize=3)
tidu_value_y = cv2.Sobel(gauss_img, cv2.CV_64F, 0, 1, ksize=3)
# 合并x和y方向上的梯度值   （X **2 + y ** 2）开方
img1 = np.sqrt(np.square(tidu_value_x) + np.square(tidu_value_y))
# 梯度方向 即角度
angle = tidu_value_y / tidu_value_x  # tan值


# 3、非极大值抑制
dx, dy = img1.shape[:2]
img_yizhi = np.zeros(img1.shape)
for i in range(1, dx - 1):   # 该算法是要比较梯度图中间像素点和周围八个像素点的数值大小，因此需要去除首行和尾行
    for j in range(1, dy - 1): # 去除首列和尾列
        flag = True  # 在8邻域内是否要抹去做个标记
        temp = img1[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵 永远是在梯度图中心点的上一行i-1和下一行i+1, 左一列j-1和右一列j+1
        # 画tan图  然后用线性插值 来判断用哪些点算
        if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (img1[i, j] > num_1 and img1[i, j] > num_2):
                flag = False

        elif angle[i, j] >= 1:
            num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if not (img1[i, j] > num_1 and img1[i, j] > num_2):
                flag = False

        elif angle[i, j] > 0:
            num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            if not (img1[i, j] > num_1 and img1[i, j] > num_2):
                flag = False

        elif angle[i, j] < 0:
            num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if not (img1[i, j] > num_1 and img1[i, j] > num_2):
                flag = False

        if flag:
            img_yizhi[i, j] = img1[i, j]

# 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
lower_boundary = img1.mean() * 0.5
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




cv2.imshow('gauss_img', img_yizhi)
cv2.waitKey(0)
