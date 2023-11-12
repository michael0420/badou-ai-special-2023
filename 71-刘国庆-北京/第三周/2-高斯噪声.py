import random
import cv2
import numpy as np


# 定义一个函数，用于给输入图像添加高斯噪声
# 参数src为输入图像，means和sigma表示正态分布的均值和标准差，percentage表示噪声所占的比例
def GaussianNoise(src, means, sigma, percentage):
    # 将输入图像赋值给NoiseImg变量，以便进行噪声处理
    NoiseImg = src
    # 计算应该添加噪声的像素数量，基于输入图像的百分比
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    # 循环NoiseNum次，即为图像的一部分像素添加噪声
    for i in range(NoiseNum):
        # 随机选择一个像素点，randX代表随机生成的行，randY代表随机生成的列
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        # 在原始像素值上加上符合正态分布的随机数，模拟高斯噪声
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        # 如果像素值小于0，则强制为0；如果像素值大于255，则强制为255，确保像素值在合理范围内
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    # 返回添加噪声后的图像
    return NoiseImg


# 读取名为 "lenna.png" 的原始彩色图像
img = cv2.imread("lenna.png", 1)
# 将原始彩色图像转换为灰度图像
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 读取名为 "lenna.png" 的灰度图像
img = cv2.imread("lenna.png", 0)
# 使用定义的GaussianNoise函数为图像添加高斯噪声，均值为2，标准差为4，噪声占比为0.8
img_gauss = GaussianNoise(img, 2, 4, 0.8)
# 在同一窗口中显示原始灰度图像和加高斯噪声后的图像，np.hstack用于将两幅图像水平堆叠
cv2.imshow("GaussianNoise", np.hstack([img_gray, img_gauss]))
# 等待用户按下任意键后关闭显示窗口
cv2.waitKey(0)
