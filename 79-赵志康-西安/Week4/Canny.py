import cv2
import numpy as np

# 读取图片
img = cv2.imread('123.png')

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 应用Canny算法
edges = cv2.Canny(gray, 50, 150)

# 显示结果
cv2.imshow('Edges', edges)
cv2.imshow('Gray',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()