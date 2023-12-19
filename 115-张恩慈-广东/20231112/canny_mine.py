# 作业2：实现canny算法
# 步骤：
# 1、将图像灰度化
# 2、将灰度化图像进行高斯滤波（过滤图像中叠加的高频噪声）
# 3、检测图像中的水平、垂直和对角边缘（Prewitt、Sobel算子等）
# 4、对梯度幅值进行非极大值抑制
# 5、用双阈值算法检测和连接边缘
import cv2
import numpy as np


# 1、将图像灰度化
def grayscale():
    src = cv2.imread('lenna.png') #BGR
    h, w = src.shape[:2]
    gray = np.zeros([h, w], src.dtype)
    for i in range(h):
        for j in range(w):
            gray[i,j] = int(0.11*src[i,j][0] + 0.59*src[i,j][1] + 0.3*src[i,j][2])
    print("灰度化结果:\n", gray)
    cv2.imshow("grayscale", gray)
    return gray

# 2、进行高斯滤波
def gaussfilter(gray):
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    dim = int(np.round(6 * sigma + 1))  # round是四舍五入函数，求高斯滤波的维度，通常是6*sigma+1
    if dim % 2 == 0:  # 将维度转成奇数，因为核的中心位置需要对齐到像素的中心位置
        dim += 1
    gauss = cv2.GaussianBlur(gray, (dim, dim), sigma)  # 高斯滤波
    cv2.imshow("gaussfilter", gauss)
    return gauss

# 3、检测图像中水平、垂直和对角边缘
def checkVerticalHorizontal(gauss):
    # 使用Sobel算子进行边缘检测
    sobelx = cv2.Sobel(gauss, cv2.CV_64F, 1, 0, ksize=3)  # x方向的边缘检测
    sobely = cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=3)  # y方向的边缘检测

    # 计算梯度幅值和方向
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    gradient_direction = np.arctan2(sobely, sobelx)

    # 不会写非极大值抑制...

if __name__ == '__main__':
    gray = grayscale()
    gauss = gaussfilter(gray)
    checkVerticalHorizontal(gauss)
    cv2.waitKey(0)