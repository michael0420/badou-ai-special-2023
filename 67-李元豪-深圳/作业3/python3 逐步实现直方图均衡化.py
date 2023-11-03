# -*- coding: utf-8 -*-  
__author__ = '李元豪 from https://www.zhilu.space'

# 导入cv2库，cv2是OpenCV的库，OpenCV是一个开源的计算机视觉和机器学习软件库
import cv2
# 导入numpy库，numpy是Python的一个强大的科学计算库，提供了大量的数学函数以及高效的多维数组对象
import numpy as np

# 使用cv2的imread方法读取图像，并将其保存到img变量中
img = cv2.imread (r"D:\AI1026\a22.png")  # 读图

# 将彩色图像转化为灰度图像
# 转为灰色图片
img_gray = cv2.cvtColor (img , cv2.COLOR_RGB2GRAY)
# 使用imshow方法显示转化后的灰度图像，并设置窗口名为"img_gray"
cv2.imshow ("img_gray" , img_gray)
# 使用waitKey方法，等待用户按键并返回按键的ASCII码，这里没有参数，会无限循环等待用户按键
cv2.waitKey ()

# 初始化一个长度为256的列表，用于存储灰度值及其对应的频度，这里初始化为0.0，代表没有频度
gray_store = [0.0] * 256
# 初始化一个长度为256的列表，用于存储灰度值及其对应的累积频度
pr_store = [0] * 256

# 获取灰度图像的尺寸，并记录灰度值的频度
# 获取图像尺寸并记录直方图列表
sp = img_gray.shape
height = sp[0]
width = sp[1]
# print(height, width)  # 打印图像的高度和宽度，为了调试用，注释掉了这一行
for i in range (height) :  # 遍历图像的所有行
    for j in range (width) :  # 遍历图像的所有列
        # 获取当前像素的灰度值，并在gray_store中找到对应的频度加1
        gray_store[img_gray[i][j]] += 1

    # 生成频度列表及累加列表
for i in range (len (gray_store)) :  # 遍历所有灰度值
    if i == 0 :  # 如果当前灰度值是0
        pr_store[i] = (gray_store[i] * 255) / (height * width)  # 则直接计算其累积频度并保存到pr_store中
    else :  # 如果当前灰度值不是0
        pr_store[i] = (gray_store[i] * 255) / (height * width) + pr_store[
            i - 1]  # 则计算其累积频度并累加到前一个灰度值的累积频度上，然后保存到pr_store中

# 均衡化操作，主要是根据累积频度来重新映射灰度值
# 均衡化操作
for i in range (len (pr_store)) :  # 遍历所有灰度值的累积频度
    k = 0  # 初始化一个新的累积频度k
    while pr_store[i] - k > 0 :  # 当当前累积频度减去k大于0时，继续循环
        k += 1  # k加1，相当于增加新的累积频度
    chazhi = k - pr_store[i]  # 计算新的累积频度和原来的累积频度的差值，即新的灰度值和原来的灰度值的差值
    if chazhi <= 0.5 :  # 如果差值小于等于0.5
        pr_store[i] = k  # 则将新的累积频度赋值给原来的累积频度，相当于增加新的灰度值
    else :  # 如果差值大于0.5
        pr_store[i] = k - 1  # 则将新的累积频度减去1赋值给原来的累积频度，相当于减少新的灰度值（因为差值已经大于0.5了）

# 根据直方图改变图像，即将灰度值映射为新的灰度值
# 根据直方图改变图像
for i in range (height) :  # 遍历所有行
    for j in range (width) :  # 遍历所有列
        img_gray[i][j] = pr_store[img_gray[i][j]]  # 将原来的灰度值映射为新的灰度值并保存到原
# 显示图片
cv2.imshow("img_gray",img_gray)
cv2.waitKey()