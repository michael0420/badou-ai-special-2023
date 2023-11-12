import random
import cv2
import numpy as np


# 定义一个名为function的函数，用于在输入图像上添加椒盐噪声
# 参数src为输入图像，percentage表示噪声所占的比例
def function(src, percentage):
    # 将输入图像赋值给NoiseImg变量，以便进行噪声处理
    NoiseImg = src
    # 计算应该添加噪声的像素数量，基于输入图像的百分比
    NOiseNum = int(percentage * src.shape[0] * src.shape[1])
    # 循环NoiseNum次，即为图像的一部分像素添加噪声
    for i in range(NOiseNum):
        # 随机选择一个像素点，randX代表随机生成的行，randY代表随机生成的列
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    # 返回添加噪声后的图像
    return NoiseImg


# 读取名为 "lenna.png" 的原始彩色图像
img = cv2.imread("lenna.png", 1)
# 将原始彩色图像转换为灰度图像
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 读取名为 "lenna.png" 的灰度图像
img = cv2.imread("lenna.png", 0)
# 使用定义的function函数为图像添加椒盐噪声，噪声占比为0.2
img_jy = function(img, 0.2)
# 在同一窗口中显示原始灰度图像和加椒盐噪声后的图像，np.hstack用于将两幅图像水平堆叠
cv2.imshow("jiaoyanzaosheng", np.hstack([img_gray, img_jy]))
# 等待用户按下任意键后关闭显示窗口
cv2.waitKey(0)
