import cv2
import matplotlib.pyplot as plt
import numpy as np


# 读取原始图
ImageSrc = cv2.imread("lenna.png", 1)
# 获取灰度图像
Imge_gray = cv2.cvtColor(ImageSrc, cv2.COLOR_BGR2GRAY)
# 直方图均衡化
Imge_gray_equ = cv2.equalizeHist(Imge_gray)
# 计算均衡化后图像的直方图
hist = cv2.calcHist([Imge_gray_equ], [0], None, [256], [0, 256])
# 绘制直方图
plt.figure()
plt.hist(Imge_gray_equ.ravel(), 256)
plt.show()
cv2.imshow("Histogram Equalization", np.hstack([Imge_gray, Imge_gray_equ]))
cv2.waitKey(0)

# 读取原始图
ImageSrc = cv2.imread("lenna.png", 1)
# 获得BGR通道的数据
(B, G, R) = cv2.split(ImageSrc)
# RGB通道进行直方图均衡化处理
R_H = cv2.equalizeHist(R)
G_H = cv2.equalizeHist(G)
B_H = cv2.equalizeHist(B)
# 将处理后的蓝色、绿色和红色通道重新合并为彩色图像
Imge_RGB = cv2.merge(B_H, G_H, R_H)
# 绘制直方图
plt.figure()
plt.hist(Imge_RGB.ravel(), 256)
plt.show()
cv2.imshow("Histogram Equalization", np.hstack([ImageSrc, Imge_RGB]))
cv2.waitKey(0)