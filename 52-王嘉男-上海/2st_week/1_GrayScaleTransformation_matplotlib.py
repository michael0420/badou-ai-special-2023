import numpy as np
import matplotlib.pyplot as plt

# 读取原始图
ImageSrc = plt.imread("hi.jpg")
# 子图位置
plt.subplot(221)
# 显示原始图
plt.imshow(ImageSrc)
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
# 子图位置
plt.subplot(222)
# 显示灰度化的图片
plt.imshow(Image_gray, cmap='gray')
# 图片关闭才结束运行程序
plt.waitforbuttonpress()
