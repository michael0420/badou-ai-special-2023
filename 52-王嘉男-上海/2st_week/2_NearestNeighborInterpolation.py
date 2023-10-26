import numpy as np
import matplotlib.pyplot as plt


# 最邻近插值函数将图片放大2倍
def function(image):
    # 获得原始图的高度,宽度和通道
    height, width, channels = image.shape
    image_empty = np.zeros(((height * 2), (height * 2), channels), np.uint8)
    sh = (height * 2) / height
    sw = (height * 2) / width
    for h in range(height * 2):
        for w in range(height * 2):
            height_new = int(h / sh)
            width_new = int(w / sw)
            image_empty[h, w] = image[height_new, width_new]
    return image_empty


# 读取原始图
ImageSrc = plt.imread("hi.jpg")
# 子图位置
plt.subplot(221)
# 显示原始图
plt.imshow(ImageSrc)
# 带入最邻近插值函数将图片放大2倍
ImageZoom = function(ImageSrc)
# 子图位置
plt.subplot(222)
# 显示缩放后的图
plt.imshow(ImageZoom)
# 图片关闭才结束运行程序
plt.waitforbuttonpress()
