import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
# 读取原始图
ImageSrc = plt.imread("hi.jpg")
# 子图位置
plt.subplot(221)
# 显示原始图
plt.imshow(ImageSrc)
# 图片灰度化
Image_gray = rgb2gray(ImageSrc)
# 子图位置
plt.subplot(222)
# 显示灰度化的图片
plt.imshow(Image_gray, cmap='gray')
# 图片二值化
Image_binary = np.where(Image_gray >= 0.5, 1, 0)
# 子图位置
plt.subplot(223)
# 显示二值化的图片
plt.imshow(Image_binary, cmap='gray')
# 图片关闭才结束运行程序
plt.waitforbuttonpress()
