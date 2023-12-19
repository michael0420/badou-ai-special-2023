#!/usr/bin/env python
# encoding=utf-8
import cv2


img = cv2.imread("./img/photo1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#  平滑噪声
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
# 膨胀操作，扩展图片中的高亮区域（白色区域），使其变得更大
#       src: 输入图像 uint8 或 float32
#       kernel：膨胀操作的结构元素，指定膨胀的形状和大小
#       dst: 可选，输出图象，不指定咋建立一个与输入图像相同尺寸和数据类型的输出图像
#       ...
dilate = cv2.dilate(gaussian, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
# 边缘检测
#  Canny
#       src：输入图像
#       threshold1: 低阈值
#       threshold2: 高阈值
#       apertureSize: sobel算子孔径大小
edge = cv2.Canny(dilate, 30, 120, apertureSize=3)
# 轮廓检测
cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0]
res = None
if len(cnts) > 0:
    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
    # print(cnts_sorted[0])
    for c in cnts_sorted:
        arcLength = cv2.arcLength(c, True)
        # print(arcLength)
        vertices = cv2.approxPolyDP(c, 0.02 * arcLength, True)
        if len(vertices) == 4:
            res = vertices
            break

for vertex in vertices:
    vertex = vertex[0]
    cv2.circle(img, tuple(vertex), 10, (255, 0, 0))
print(res)
cv2.imshow('vertices', img)
cv2.waitKey(0)
