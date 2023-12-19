import cv2
import matplotlib.pyplot as plt


img = cv2.imread("img/lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.calcHist()
# 计算图像直方图
#       image: [img] 用于计算直方图的图像列表
#       channels: [0, 1, 2] 指定计算的通道（BGR）
#       mask: 掩码 指定用于计算直方图的图像区域，全图直接设置None，指定区域需要一个与原图相同尺寸的二值图像
#       histSize: [256] 指定直方图的大小(柱子的数量)
#       ranges: [0, 256] 指定像素值的范围
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title("Gray Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.xlim([0, 256])
plt.plot(hist)
plt.show()
