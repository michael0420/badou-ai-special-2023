# 计算直方图
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

# 读取灰度图像
# gray_img = cv.imread('lenna.png',cv.IMREAD_GRAYSCALE)

img = cv.imread('lenna.png')
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# print(img.shape[:2])

# cv.imshow('img',gray_img)
'''
# 计算灰度直方图 方式1，使用plt
plt.figure()
plt.hist(gray_img.ravel(),256)
plt.show()
'''
'''
# 计算灰色直方图 方式2
hist = cv.calcHist([gray_img],[0],None,[256],[0,256])
'''
'''
# 画图
plt.figure()
plt.title('gray hist')
plt.xlabel('gray')
plt.ylabel('num')
plt.plot(hist)
plt.xlim([0,256])
plt.show()
'''

'''
# 彩色直方图
chans = cv.split(img)
print(chans)

colors = ('b','g','r')

plt.figure()
plt.title('color hist')
plt.xlabel('color')
plt.ylabel('num')
plt.xlim([0,256])

for chan,color in zip(chans,colors):
    color_hist = cv.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(color_hist,color=color)

plt.show()
'''

# 直方图均衡化
# equ = cv.equalizeHist(gray_img)

'''
print(equ)

plt.figure()
plt.title('hist equ')
plt.xlabel('gray')
plt.ylabel('num')
plt.xlim([0,256])

plt.hist(equ.ravel(),256)

plt.show()

cv.imwrite('hist_equ.png',equ)
cv.imwrite('gray.png',gray_img)
'''


def equHist(img):
    h, w = img.shape
    # hist = cv.calcHist([img],[0],None,[256],[0,256])

    returnImg = np.zeros((h, w), dtype=np.uint8)

    hist = {}
    for i in range(h):
        for j in range(w):

            if img[i, j] not in hist:
                hist[img[i, j]] = 1
            else:
                hist[img[i, j]] += 1


    a = sorted(hist.items(), key=lambda x: x[0])
    hist = dict(a)
    sum_q = 0
    ys = {}
    for color, num in hist.items():
        sum_q += hist[color] / (h * w)
        ys[color] = round(sum_q * 256 - 1)
        # if n <= 100:
        # print(returnImg[i])

    for i in range(h):
        for j in range(w):
            returnImg[i, j] = ys[img[i, j]]
    return returnImg


equ_hist = equHist(gray_img)
cv.imwrite('equ_hist2.png', equ_hist)
