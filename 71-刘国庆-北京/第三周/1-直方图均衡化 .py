import cv2
import matplotlib.pyplot as plt
import numpy as np

# 灰色图像直方图均衡化
# 从文件 "lenna.png" 中以彩色图像方式读取图像数据
img = cv2.imread("lenna.png", 1)
# 将彩色图像转换为灰度图像
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 灰度图像直方图均衡化
img_equ = cv2.equalizeHist(img_gray)
# 计算均衡化后图像的直方图
hist = cv2.calcHist([img_equ], [0], None, [256], [0, 256])
# 创建一个新的图形窗口
plt.figure()
# 绘制直方图，img_equ.ravel()将二维数组转换为一维数组，256表示直方图的bin数量
plt.hist(img_equ.ravel(), 256)
# 显示绘制的直方图
plt.show()
# 在同一窗口中显示原始灰度图像和均衡化后的图像，np.hstack用于将两幅图像水平堆叠
cv2.imshow("Histogram Equalization", np.hstack([img_gray, img_equ]))
# 等待用户按下任意键后关闭窗口
cv2.waitKey(0)

# 彩色图像直方图均衡化
# 从文件 "lenna.png" 中以彩色图像方式读取图像数据
img = cv2.imread("lenna.png", 1)
# 分离原始图像的蓝色、绿色和红色通道数据
(b, g, r) = cv2.split(img)
# 对蓝、绿、红通道进行直方图均衡化处理，增强图像对比度
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)
# 将处理后的蓝色、绿色和红色通道重新合并为彩色图像
img_rgb = cv2.merge((bh, gh, rh))
# 创建一个新的图形窗口
plt.figure()
# 绘制直方图，img_rgb.ravel()将二维数组转换为一维数组，256表示直方图的bin数量
plt.hist(img_rgb.ravel(), 256)
# 显示绘制的直方图
plt.show()
# 在同一窗口中显示原始灰度图像和均衡化后的图像，np.hstack用于将两幅图像水平堆叠
cv2.imshow("Histogram Equalization", np.hstack([img, img_rgb]))
# 等待用户按下任意键，然后关闭所有窗口
cv2.waitKey(0)
