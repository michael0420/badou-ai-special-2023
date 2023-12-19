import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png')
plt.subplot(221)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(222)
chans = cv2.split(img)
colors = ("b", "g", "r")
plt.xlabel("bin")
plt.ylabel("pix")
plt.title("color hist")

for(chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
    plt.xlim([0, 255])

#3通道彩色图均衡化
plt.subplot(223)
bH = cv2.equalizeHist(chans[0])
gH = cv2.equalizeHist(chans[1])
rH = cv2.equalizeHist(chans[2])
result_img = cv2.merge((rH, gH, bH))
plt.imshow(result_img)

# 彩色直方图均衡化之后的直方图
plt.subplot(224)
chans = cv2.split(result_img)
colors = ("b","g","r")
plt.xlabel("bin")
plt.ylabel("# of pix")
plt.title("color hist")

for(chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0, 256])
    plt.plot(hist, color = color)
    plt.xlim([0,255])

plt.show()
