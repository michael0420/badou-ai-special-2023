import cv2
import numpy as np

img = cv2.imread('photo.jpg')
# 获取灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 采用高斯滤波去除噪点
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#getStructuringElement 返回指定形状和尺寸的结构元素 MORPH_RECT 矩形  MORPH_CROSS 交叉型
#  dilate 将边界的白色像素生长扩大，增强边界的链接，具体是只要周围窗口有白色点，新图像的像素就是白色
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
# 用canny 做边缘检测。
edged = cv2.Canny(dilate, 30, 120, 3)
# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0]
#if imutils.is_cv2() else cnts[1]  # 判断是opencv2还是opencv3
docCnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # 根据轮廓面积从大到小排序
    for c in cnts:
        peri = cv2.arcLength(c, True)                           # 计算轮廓周长
        approx = cv2.approxPolyDP(c, 0.02*peri, True)           # 轮廓多边形拟合
        # 轮廓为4个点表示找到纸张
        if len(approx) == 4:
            docCnt = approx
            break

for peak in docCnt:
    peak = peak[0]
    cv2.circle(img, tuple(peak), 10, (255, 0, 0))

cv2.imshow('img', img)
cv2.waitKey(0)