import  cv2
import numpy as np
from matplotlib import pyplot as plt

#获取图像直方图
img = cv2.imread('lenna.png',1)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #将图片转成灰度图
# cv2.imshow("image_gray",gray)
# cv2.waitKey(0)

#获取直方图,方法一
# plt.figure()
# plt.hist(gray.ravel(),256)
# plt.show()

#直方图均衡化
# dst = cv2.equalizeHist(gray)
# plt.hist(dst.ravel(),256)
# plt.show()

#分别显示原图，和均衡化之后的图
# hist = cv2.calcHist([dst],[0],None,[256],[0,256])
# cv2.imshow("Histogram Equalization",np.hstack([gray, dst]))
# cv2.waitKey(0)

#彩色多通道均衡化
cv2.imshow("src",img)

(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

result = cv2.merge((bH,gH,rH))
cv2.imshow("dst",result)
cv2.waitKey(0)

#获取直方图，方法二
'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''
# hist = cv2.calcHist([gray],[0],None,[256],[0,256])
# plt.figure() #新建图像
# plt.title("Grayscale Histogram") #添加标题
# plt.xlabel("Bins")#x轴标签
# plt.ylabel('# of Pixels') #y轴标签
# plt.plot(hist)
# plt.xlim([0,256]) #设置x坐标轴范围
# plt.show()

#彩色直方图
# cv2.imshow("original",img)
# cv2.waitKey(0)

# chans = cv2.split(img)
# colors = ["b","g","r"]
# plt.figure()
# plt.title("Flattened Color Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
#
# for (chan,color) in zip(chans,colors):
#     hist = cv2.calcHist([chan],[0],None,[256],[0,256])
#     plt.plot(hist,color = color)
#     plt.xlim([0,256])
# plt.show()

