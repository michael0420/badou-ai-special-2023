import random
import cv2
import numpy as np


# image_src为输入图像，means和sigma表示正态分布的均值和标准差，percentage表示噪声所占的比例
def GaussianNoise(image_src, means, sigma, percentage):
    # 将输入图像赋值给NoiseImg变量，以便进行噪声处理
    imge_noise = image_src
    # 计算应该添加噪声的像素数量，基于输入图像的百分比
    noise_cnt = int(percentage * ImageSrc.shape[0] * ImageSrc.shape[1])
    # 循环NoiseNum次，即为图像的一部分像素添加噪声
    for i in range(noise_cnt):
        # 随机选择一个像素点，randX代表随机生成的行，randY代表随机生成的列
        rand_x = random.randint(0, ImageSrc.shape[0] - 1)
        rand_y = random.randint(0, ImageSrc.shape[1] - 1)
        # 在原始像素值上加上符合正态分布的随机数，模拟高斯噪声
        imge_noise[rand_x, rand_y] = imge_noise[rand_x, rand_y] + random.gauss(means, sigma)
        # 如果像素值小于0，则强制为0；如果像素值大于255，则强制为255，确保像素值在合理范围内
        if imge_noise[rand_x, rand_y] < 0:
            imge_noise[rand_x, rand_y] = 0
        elif imge_noise[rand_x, rand_y] > 255:
            imge_noise[rand_x, rand_y] = 255
    # 返回添加噪声后的图像
    return imge_noise


# 读取原始图
ImageSrc = cv2.imread("lenna.png", 1)
# 将原始彩色图像转换为灰度图像
Imge_gray = cv2.cvtColor(ImageSrc, cv2.COLOR_BGR2GRAY)

# 读取原始图
ImageSrc = cv2.imread("lenna.png", 0)
# 添加高斯噪声，均值为2，标准差为4，噪声占比为0.8
Imge_gauss = GaussianNoise(ImageSrc, 2, 4, 0.8)

# 显示结果
cv2.imshow("Gaussi Noise", np.hstack([Imge_gray, Imge_gauss]))
cv2.waitKey(0)