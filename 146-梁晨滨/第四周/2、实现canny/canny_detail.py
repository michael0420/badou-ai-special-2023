import numpy as np
from matplotlib import pyplot as plt
import math

plt.rcParams['font.sans-serif'] = [u'SimHei']


def Canny(img, sigma, low, high):
    # 灰度化
    H, W, C = img.shape
    img_gray = np.zeros([H, W])
    img_gray[:, :] = img[:, :, 0] * 0.11 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.3

    # 1、高斯平滑
    # Gaussion_kernel（dim x dim） 的卷积核，需要为奇数
    dim = int(np.round(6 * sigma + 1))
    if dim % 2 == 0:
        dim += 1
    Gaussion_kernel = np.zeros([dim, dim])


    tmp = [i - dim // 2 for i in range(dim)]
    n1 = 1 / (2 * math.pi * sigma ** 2)     # n1 = 1/(2πσ^2)
    n2 = -1 / (2 * sigma ** 2)              # n2 = -1/2σ^2
    for i in range(dim):
        for j in range(dim):
            # tmp[i]就是(i - x0)^2, tmp[j]就是(j - y0)^2,
            # (x0, y0)是高斯核中的坐标，例如九宫格的高斯核，
            # 每个点坐标就是(tmp[i], tmp[j])
            #       -1        0          1
            # -1 (-1, -1)  (0, -1)   (1, -1)
            #  0 (-1,  0)  (0,  0)   (1,  0)
            #  1 (-1,  1)  (0,  1)   (1,  1)
            # 这就是高斯核的公式，只不过写一起太长了，系数都用n1和n2替代了,(i,j)需要转化为相对坐标(tmp[i], tmp[j])
            # 最终高斯公式(i, j) = 1/(2πσ^2)* e ^ -(tmp[i] ^ 2 + tmp[j] ^ 2)/2σ^2
            Gaussion_kernel[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    # 图像除平均值，平滑
    Gaussion_kernel = Gaussion_kernel / Gaussion_kernel.sum()

    # 对原图像进行padding，防止卷积后图像缩小
    img_filter = np.zeros([H, W])
    padding = dim // 2
    # print(padding, img_gray.shape)
    img_padding = np.pad(img_gray, ((padding, padding), (padding, padding)), 'constant')
    for y in range(H):
        for x in range(W):
            # img_filter是高斯平滑后的图像
            img_filter[y, x] = np.sum(img_padding[y: y + dim, x: x + dim] * Gaussion_kernel)


    # 2、求梯度
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_sobel_x = np.zeros(img_padding.shape)
    img_tidu_sobel_y = np.zeros(img_padding.shape)
    img_tidu_sobel_xy = np.zeros(img_padding.shape)
    # 对高斯后的图像求梯度，高斯图像填充防止卷积后变小
    img_padding = np.pad(img_filter, ((1, 1), (1, 1)), 'constant')

    for i in range(H):
        for j in range(W):
            img_tidu_sobel_x[i, j] = np.sum(img_padding[i: i + 3, j: j + 3] * sobel_x)
            img_tidu_sobel_y[i, j] = np.sum(img_padding[i: i + 3, j: j + 3] * sobel_y)
            # 最终梯度是水平和竖直方向梯度的平方根
            img_tidu_sobel_xy[i, j] = np.sqrt(img_tidu_sobel_x[i, j] ** 2 + img_tidu_sobel_y[i, j] ** 2)

    # 计算角度angle=（y/x）,x是0会报错，所以设为无穷小值
    img_tidu_sobel_x[img_tidu_sobel_x == 0] = 0.000000001
    angle = img_tidu_sobel_y / img_tidu_sobel_x

    # 3、非极大值抑制
    img_nms = np.zeros(angle.shape)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            flag = True
            tidu = img_tidu_sobel_xy[i - 1: i + 2, j - 1: j + 2]
            if angle[i, j] <= -1:
                num1 = (tidu[0, 1] - tidu[0, 0]) / angle[i, j] + tidu[0, 1]
                num2 = (tidu[2, 1] - tidu[2, 2]) / angle[i, j] + tidu[2, 1]
                if img_tidu_sobel_xy[i, j] <= max(num1, num2):
                    flag = False
            elif angle[i, j] >= 1:
                num1 = (tidu[0, 2] - tidu[0, 1]) / angle[i, j] + tidu[0, 1]
                num2 = (tidu[2, 0] - tidu[2, 1]) / angle[i, j] + tidu[2, 1]
                if img_tidu_sobel_xy[i, j] <= max(num1, num2):
                    flag = False
            elif angle[i, j] > 0:
                num1 = (tidu[0, 2] - tidu[1, 2]) / angle[i, j] + tidu[1, 2]
                num2 = (tidu[2, 0] - tidu[1, 0]) / angle[i, j] + tidu[1, 0]
                if img_tidu_sobel_xy[i, j] <= max(num1, num2):
                    flag = False
            elif angle[i, j] < 0:
                num1 = (tidu[1, 0] - tidu[0, 0]) / angle[i, j] + tidu[1, 0]
                num2 = (tidu[1, 2] - tidu[2, 2]) / angle[i, j] + tidu[1, 2]
                if img_tidu_sobel_xy[i, j] <= max(num1, num2):
                    flag = False

            if flag:
                img_nms[i, j] = img_tidu_sobel_xy[i, j]

    # 4、双阈值限制
    threshold_low = img_tidu_sobel_xy.mean() * low
    threshold_high = img_tidu_sobel_xy.mean() * high
    bian = []
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if img_nms[i, j] >= threshold_high:
                img_nms[i, j] = 255
                bian.append([i, j])
            elif img_nms[i, j] <= threshold_low:
                img_nms[i, j] = 0

    while bian:
        x, y = bian.pop()
        for i in range(x - 1, x + 3):
            for j in range(y - 1, y + 3):
                if (img_nms[i, j] > threshold_low) and (img_nms[i, j] <= threshold_high):
                    img_nms[i, j] = 255
                    bian.append([i, j])
    for i in range(H):
        for j in range(W):
            if (img_nms[i, j] != 0) and (img_nms[i, j] != 255):
                img_nms[i, j] = 0



    return img_nms


if __name__ == '__main__':
    img_input = plt.imread("lenna.png")
    img_input = (img_input * 255).astype(np.uint8)

    sigma, low, high = 0.5, 0.5, 1.5

    img_canny = Canny(img_input, sigma, low, high)

    plt.subplots_adjust(hspace=0.5)
    plt.subplot(121), plt.title("原图")
    plt.imshow(img_input)

    plt.subplot(122), plt.title("Canny检测后的边缘图")
    plt.imshow(img_canny, cmap='gray')
    plt.show()