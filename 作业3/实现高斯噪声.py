# 导入cv2库，cv2是OpenCV的库，OpenCV是一个开源的计算机视觉和机器学习软件库  
import cv2

# 导入os库，os库提供了很多与操作系统交互的函数，例如访问文件路径、创建文件夹等  
import os

# 导入numpy库，numpy是Python的一个数学库，提供大量的数学函数以及矩阵运算的功能  
import numpy as np


# 定义一个函数add_noise_Guass，该函数的作用是为图像添加高斯噪声
def add_noise_Guass(img , mean=0 , var=0.01) :
    # 将图像数据类型转换为浮点数，并归一化到[0,1]之间  
    img = np.array (img / 255 , dtype=float)

    # 生成与图像大小相同的高斯噪声，高斯噪声的平均值为mean，方差为var  
    noise = np.random.normal (mean , var ** 0.5 , img.shape)

    # 将噪声添加到图像上  
    out_img = img + noise

    # 如果处理后的图像有像素值小于0，那么将low_clip设为-1，否则设为0  
    if out_img.min () < 0 :
        low_clip = -1
    else :
        low_clip = 0

        # 对图像进行裁剪，确保所有像素值都在0-1之间
    out_img = np.clip (out_img , low_clip , 1.0)

    # 将图像数据类型转换回uint8，并归一化到[0,255]之间  
    out_img = np.uint8 (out_img * 255)

    # 返回处理后的图像  
    return out_img

img1 = cv2.imread (r"D:\AI1026\a22.png")
out_img = add_noise_Guass (img1)

        # 显示处理后的图像  
cv2.imshow ("img" , out_img)

        # 等待用户按键后关闭显示窗口  
cv2.waitKey (0)


