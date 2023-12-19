# -*- coding=utf-8
import cv2
import numpy as np

"""
寻找顶点
"""
#读取图片，转化为灰度图
img = cv2.imread('photo1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#高斯滤波
blurred = cv2.GaussianBlur(gray,(5,5),0)

#形态学膨胀
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))

#边缘检测
edged = cv2.Canny(dilate, 30 ,120,3)

#轮廓检测
cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#返回两个值：检测到的轮廓列表+图像层次结构
cnts = cnts[0]
docCnt = None

#对轮廓处理，检测最优
if len(cnts) > 0:
    #按轮廓构成的面积从大到小排序
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
    for c in cnts:
        #计算轮廓周长
        peri = cv2.arcLength(c,True)
        #多边形逼近，多边形闭合
        approx = cv2.approxPolyDP(c, 0.02*peri,True)
        #轮廓四个点为找到纸张
        if len(approx) == 4:
            docCnt = approx
            break


src =[]
#在图像上绘制圆圈，看是否找到顶点
for peak in docCnt:
    #确保顶点坐标为二维点非数组
    peak = peak[0]
    src.append(peak)
    cv2.circle(img, tuple(peak), 10, (0,255,0))
#绘制图才能显现圆圈
cv2.imshow('img',img)
cv2.waitKey(0)

#src
#[[207 151]
# [ 16 603]
# [344 732]
# [518 283]]

"""
透视变换
"""
src = np.float32(src)
dst = np.float32([[0,0],[0,488],[337,488],[337,0]])
print(src)
#生成透视变换矩阵
m = cv2.getPerspectiveTransform(src,dst)
print('warpMatrix:\n',m)

#作图像的透视变换
result = cv2.warpPerspective(img.copy(), m, (337,488))
cv2.imshow('result',result)
cv2.waitKey(0)

