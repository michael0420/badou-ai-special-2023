import cv2
import matplotlib.pyplot as plt

# imread
# filename: const string
# flags: int
#       -1: 已废弃
#       0: 返回灰色图像
#       1: 默认值，返回彩色图像
#       2: 图像颜色深度为16或32时返回对应深度，否则返回8位深度
#       4： 返回所有颜色
#       128： 忽略任何旋转
#       若希望载入最真实的图片，应选择2或4，或直接2|4
img = cv2.imread("img/lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# hist = cv2.calcHist(gray, 0, None, [256], [0, 256])
# plt.figure()
# 用于创建一个新的图形窗口，可以控制图形大小，分辨率等
#       figsize: 设置图形窗口大小, 如(6, 4)， 单位英寸
#       dpi: 设置分辨率
# print(gray)
plt.figure(figsize=(10, 5))
# plt.hist()
# 用于绘制直方图
#       data: 一维数据（数组，列表，元组）
#       bins: int|一维数据 整数表示设置等宽序列个数，一维数据表示范围，[0， 3，5]数据表示分两组，0-3和3-5
#       range: 指定数据范围，仅显示数据范围内的数据
#       density: bool 设置True时显示频率分布
#       color: string 直方图颜色
#       alpha: int|float 直方图透明度
# ravel()
#       数组.ravel() 将多维数组拉平为一维数组，建立原数组的视图
#       修改返回的数组会影响原始数组
#       若要创建副本，应使用flatten
plt.hist(gray.ravel(), 256)
plt.show()
