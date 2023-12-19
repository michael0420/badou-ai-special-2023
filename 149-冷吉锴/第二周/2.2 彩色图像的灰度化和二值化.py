import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray

# 灰度化 方法1（按公式浮点算法）
img = cv2.imread("lenna.png")
h, w = img.shape[:2]  # 获取图片的high和wide 行和列
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 获取当前像素值，取出当前high和wide中的BGR坐标 (三通道)
        img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像
print(img.shape)
print(img)  # 打印原始图片
print("image show gray: %s" % img_gray)  # 打印灰度图
cv2.imshow("image show gray", img_gray)

# 画出原图
plt.subplot(331)  # 创建子图位于图形窗口的左上角
img = plt.imread("lenna.png")
plt.imshow(img)  # 当前激活的子图中显示图像
print("---image lenna---")
print(img)  # 打印图片的像素数据，前两维是high和wide，第三维度是颜色通道

# 画出方法1的灰度图
plt.subplot(332)  # 创建子图位于图形窗口的第一排第二个位置
plt.imshow(img_gray, cmap='gray')

# 灰度化 方法2（调用函数）
img_gray_2 = rgb2gray(img)
plt.subplot(333)  # 创建子图位于图形窗口的右上角
plt.imshow(img_gray_2, cmap='gray')
print("---image lenna gray 2---")
print(img_gray_2)  # 打印图片的像素数据，前两维是high和wide，第三维度是颜色通道

# 灰度化 方法3（调用函数）
img_gray_3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(334)  # 创建子图位于图形窗口的左下角
plt.imshow(img_gray_3, cmap='gray')

# 二值化 方法1
rows, cols = img_gray_2.shape
for i in range(rows):
    for j in range(cols):
        if img_gray_2[i, j] <= 0.5:
            img_gray_2[i, j] = 0
        else:
            img_gray_2[i, j] = 1
plt.subplot(335)
plt.imshow(img_gray_2, cmap='gray')

# 二值化 方法2
img_binary = np.where(img_gray_2 >= 0.5, 1, 0)
print("---image_binary---")
print(img_binary)
print(img_binary.shape)
plt.subplot(336)
plt.imshow(img_binary, cmap='gray')

plt.show()
