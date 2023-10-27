"""
@author: 67李元豪
彩色图像的灰度化、二值化
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2


# 灰度化实现方式（原理）
# cv中读取颜色为BGR而不是主流的RBG，大坑
# img1 = cv2.imread (img3)
img1 = cv2.imread (r"D:\AI1026\a22.png")
h , w = img1.shape[:2]  # 获取图片的高、宽
img2 = np.zeros ([h , w] , img1.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range (h) :
    for j in range (w) :
        # 取出当前high和wide中的BGR坐标
        m = img1[i , j]
        # 将BGR坐标转化gray坐标并赋值给新图像
        # 浮点算法：Gray = R0.3 + G0.59 + B0.11
        img2[i , j] = int (m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
print("原图三色通道显示:%s" % img1)
print("灰度图单色通道显示:%s" % img2)
cv2.imshow ("image show gray" , img2)
cv2.imshow ("image" , img1)

# 二值化实现方式（原理）
img3 = np.zeros ([h , w] , img1.dtype)
for i in range (h) :
    for j in range (w) :
        if img2[i , j] <= 128 :
            img3[i , j] = 0
        else :
            img3[i , j] = 1
# cv2.imshow()没办法把二值图正常展示出来，需要用到plt
plt.imshow (img3 , cmap='gray')  # cmap表示二值是哪二值gray为黑白
plt.show ()
cv2.waitKey (0)
