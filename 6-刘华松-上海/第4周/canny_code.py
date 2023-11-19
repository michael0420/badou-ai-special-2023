#实现 canny (detail,调接口)

#Canny边缘检测是一种常用的图像处理算法，用于检测图像中的边缘。

import cv2

# 读取图像
image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

# 进行Canny边缘检测
edges = cv2.Canny(image, threshold1=100, threshold2=200)

# 显示原始图像和边缘检测结果
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)


#注意：一般来说，较小的阈值用于检测弱边缘，较大的阈值用于检测强边缘。