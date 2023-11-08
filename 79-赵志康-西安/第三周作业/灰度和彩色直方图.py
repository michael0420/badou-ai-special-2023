import cv2
import cv2.gapi
import numpy as np
import matplotlib.pyplot as plt
'''
1、使用matplotlib.pyplot.hist(x, bins)函数绘制直方图
　　x：数据，必须是一维的。图像数据通常是二维的，所以要用ravel()函数将其拉成一维的数据后再作为参数使用。
　　bins：表示x轴分多少组。
2、使用hist = cv2.calcHist(img, channels, mask, histSize, ranges, accumulate)函数统计直方图，然后用plt.plot()函数将其绘制出来
　　这个函数里面的参数都要加［］括起来，这是底层代码写死的，不然无法识别。
　　img：原图像
　　channels：如果输入的图像是灰度图，它的值就是[0]，如果是彩色图像，传入的参数可以是[0]、[1]、[2],分布对应着b,g,r。
　　mask：掩膜图像。要统计整幅图像的直方图就把这个参数设为None，如果要统计掩膜部分的图像的直方图，就需要这个参数。
　　histSize：bins的个数,就是分多少个组距。例如[256]或者[16]，都要用方括号括起来。
　　ranges：像素值范围（横轴范围），通常为[0, 256]
　　accumulate：累计(累积、叠加)标识，默认值是False。
　　这个函数返回的对象hist是一个一维数组，数组内的元素是各个灰度级的像素个数。
calcHist—计算图像直方图
'''

##灰度图像直方图
#获取灰度图像
img = cv2.imread("123.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#灰度图像的直方图
plt.figure()

plt.subplot(321)
plt.title("Gray Histogram")     #添加直方图标题
plt.xlabel("Bins")              #添加直方图X轴描述
plt.ylabel("# of Pixels")       #添加直方图Y轴描述
plt.hist(img_gray.ravel(),256)  #ravel()函数是将一个多维数组转化为一维数组，将灰度级划分为256个等级，hist函数绘制直方图

plt.subplot(322)
plt.imshow(img_gray,cmap='gray')

#彩色图像直方图

chans = cv2.split(img_color) #将彩色图像三个通道进行分离，split函数分离通道的顺序是B、G、R
print(chans)
colors = ("b","g","r")
plt.subplot(325)
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim([0,256])        #x轴范围
plt.subplot(326)
plt.imshow(img_color)
plt.show()