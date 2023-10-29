'''
功能：实现灰度化和二值化
这段代码首先读取一张名为"lenna.png"的图片，然后将其转换为灰度图像，接着对灰度图像进行二值化处理，最后将处理后的二值化图像显示出来，可以将"lenna.png"替换为其他图片文件名来查看不同的效果。
''' 
from skimage.color import rgb2gray  # 导入skimage库中的rgb2gray函数，用于将RGB图像转换为灰度图像
import numpy as np  # 导入numpy库，用于进行数值计算和处理数组
import matplotlib.pyplot as plt  # 导入matplotlib库的pyplot模块，用于绘制图像
from PIL import Image  # 导入PIL库的Image模块，用于处理图像文件
import cv2  # 导入cv2库，用于处理图像和视频数据

# 读取图像文件
img = cv2.imread("lenna.png")
h, w = img.shape[:2]  # 获取图像的高度和宽度
img_gray = np.zeros([h, w], img.dtype)  # 创建与原图像大小相同的灰度图像
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 获取当前像素的BGR值
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR值转换为灰度值并赋值给新图像
print(img_gray)
print("image show gray: %s" % img_gray)
cv2.imshow("image show gray", img_gray)

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna----")
print(img)

# 使用skimage库的rgb2gray函数将图像转换为灰度图像
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

# 对灰度图像进行二值化处理，大于等于0.5的像素值为1，小于0.5的像素值为0
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
