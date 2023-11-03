import cv2
img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#灰度图像
cv2.imshow('Histogram', gray)
# cv2.waitKey()

#灰度直方图
#方法一：plt
import matplotlib.pyplot as plt
plt.figure()
plt.title("Grayscale Histogram")
plt.hist(gray.ravel(), 256)    #gray.ravel()将二维矩阵变成一维矩阵
#方法二：calcHist
'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Grayscale value")
plt.ylabel("Number of statistics")
plt.plot(hist)
plt.xlim([0,256])


#彩色直方图
channels = cv2.split(img)
colos = ['b', 'g', 'r']
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Value")
plt.ylabel("Number of statistics")
for (channel, color) in zip(channels, colos):
    print(channel, color)
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
    plt.xlim([0, 256])

plt.show()
