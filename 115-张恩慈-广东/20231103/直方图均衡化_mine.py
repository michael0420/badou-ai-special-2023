# 作业1：直方图均衡化

import cv2
import matplotlib.pyplot as plt
import numpy as np

#1、加载原图，默认是彩色  cv2.IMREAD_COLOR
src = cv2.imread("lenna.png")
#2、转为灰度图
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# cv2.imshow("src_gray", gray)

#3、对灰度图 进行直方图均衡化
dst = cv2.equalizeHist(gray)
# cv2.imshow("dst", dst)

# 4、定义一个横纵坐标系
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

# 5、显示 拼接直方图均衡化前图和之后图
cv2.imshow("histogram equalization", np.hstack([gray, dst]))

# 6、三通道图像直方图均衡化(分三通道)
(b, g, r) = cv2.split(src)
# 分别得到三通道直方图均衡化结果
b_dst = cv2.equalizeHist(b)
g_dst = cv2.equalizeHist(g)
r_dst = cv2.equalizeHist(r)
# 三个合并为一个
combine = cv2.merge((b_dst, g_dst, r_dst))
cv2.imshow("combine", np.hstack([src, combine]))
cv2.waitKey(0)
