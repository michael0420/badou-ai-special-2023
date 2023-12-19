import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
matplotlib是一个常用的绘图库，可以用来绘制图表、直方图、曲线图等各种可视化效果
'''

# 获取灰度图像  第二个参数0为灰度图 1为彩色图
img = cv2.imread("a.png",1)
# 将彩色图片img转换成了灰度图片gray
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 显示灰度图片的 参数1为窗口名称 参数2为要显示的灰度图
# cv2.imshow("image_gray", gray)
# cv2.waitKey()

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)
print('dst', dst)
# 计算图像直方图
'''
参数1 输入图像 必须是单通道灰度图像
参数2 直方图的通道 0单通道灰度图像
参数3 指定直方图的掩码
参数4 灰度级的范围分成多少份一般设置为256
参数5 灰度级的范围，一般为[0, 256]，表示所有可能的灰度级
'''
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
print('hist',hist)
# 使用matplotlib库创建一个新的图像窗口
plt.figure()
# 绘制直方图
'''
dst.ravel()将二维的图像数据转换成一维数组
参数1 绘制直方图的数据
参数2 指定了直方图的箱子（bin）数目 是直方图中将灰度级的范围分成多少份的参数
在直方图中，每个箱子代表了一个灰度级范围，通过统计落入每个灰度级范围内的像素数量，可以得到该灰度级范围内的像素分布情况
'''
plt.hist(dst.ravel(), 256)
plt.show()
# 这一句将经过直方图均衡化处理后的灰度图像和原始的灰度图像在同一窗口中进行对比显示。通过 np.hstack() 函数将两个图像在水平方向拼接起来，以便更好地观察它们之间的差异。
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
# 这一句则是将经过直方图均衡化处理后的彩色图像 dst 显示在名为 "dst_rgb" 的窗口中。
cv2.imshow("dst_rgb", dst)

cv2.waitKey(0)


# 彩色图像直方图均衡化
# img = cv2.imread("lenna.png", 1)
# cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
# (b, g, r) = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# 合并每一个通道
# result = cv2.merge((bH, gH, rH))
# cv2.imshow("dst_rgb", result)
# cv2.waitKey(0)
