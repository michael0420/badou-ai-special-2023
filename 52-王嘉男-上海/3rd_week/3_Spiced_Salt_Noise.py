import random
import cv2
import numpy as np


# image_src为输入图像，percentage表示噪声所占的比例
def function(image_src, percentage):
    # 将输入图像赋值给NoiseImg变量，以便进行噪声处理
    imge_noise = image_src
    # 计算应该添加噪声的像素数量，基于输入图像的百分比
    noise_cnt = int(percentage * image_src.shape[0] * image_src.shape[1])
    # 循环NoiseNum次，即为图像的一部分像素添加噪声
    for i in range(noise_cnt):
        # 随机选择一个像素点，randX代表随机生成的行，randY代表随机生成的列
        rand_x = random.randint(0, image_src.shape[0] - 1)
        rand_y = random.randint(0, image_src.shape[1] - 1)
        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() <= 0.5:
            imge_noise[rand_x, rand_y] = 0
        else:
            imge_noise[rand_x, rand_y] = 255
    # 返回添加噪声后的图像
    return imge_noise


# 读取原始图
ImageSrc = cv2.imread("lenna.png", 1)
# 将原始彩色图像转换为灰度图像
Imge_gray = cv2.cvtColor(ImageSrc, cv2.COLOR_BGR2GRAY)

# 读取原始图
ImageSrc = cv2.imread("lenna.png", 0)
# 添加椒盐噪声，噪声占比为0.2
Imge_spiced_salt_noise = function(ImageSrc, 0.3)

# 显示结果
cv2.imshow("Spiced Salt Noise", np.hstack([Imge_gray, Imge_spiced_salt_noise]))
cv2.waitKey(0)