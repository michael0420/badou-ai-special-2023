import cv2
import numpy as np

# 读取原始图
ImageSrc = cv2.imread("hi.jpg")
# cv2读取的图片的通道排列是BGR，转换为RGB
img_gray = cv2.cvtColor(ImageSrc, cv2.COLOR_BGR2GRAY)
# 显示原始图
cv2.imshow('picture Source', ImageSrc)
# 获得原始图的高度和宽度
height, width = ImageSrc.shape[:2]
# 用于储存灰度化图片的容器进行初始化
Image_gray = np.zeros([height, width], ImageSrc.dtype)
# 按照原始图的高宽进行像素点遍历
for h in range(height):
    for w in range(width):
        # 存储当前坐标的RGB值
        ImageRGB_Temp = ImageSrc[h, w]
        # 当前坐标的RGB值按照公式转化为灰度值并储存在容器中
        Image_gray[h, w] = int((ImageRGB_Temp[0] * 0.3) + (ImageRGB_Temp[1] * 0.59) + (ImageRGB_Temp[2] * 0.11))
# 显示灰度化的图片
cv2.imshow('gray picture ', Image_gray)
# 图片关闭才结束运行程序
cv2.waitKey(0)
