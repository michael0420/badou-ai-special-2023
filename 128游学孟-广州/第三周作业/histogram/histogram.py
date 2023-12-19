import cv2
import numpy as np

# 读取原始图像
image = cv2.imread(r'D:\xuexi\zuoye\week3\histogram\histogram.jpg')

target_width = int(image.shape[1]/10)
target_height = int(image.shape[0]/10)

# 下采样缩小十倍
downsampled_image = cv2.resize(image,(target_width,target_height))

# 图像处理（直方图均衡化）
gray_image = cv2.cvtColor(downsampled_image, cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(gray_image)

# 显示结果
cv2.imshow('downsampled_image', downsampled_image)
cv2.imshow('gray_image',gray_image)
cv2.imshow('equ', equ)

#保存结果
cv2.imwrite('histogramdown.jpg',downsampled_image)
cv2.imwrite('histogramgray.jpg',gray_image)
cv2.imwrite('histogramequ.jpg',equ)
cv2.waitKey(0)
cv2.destroyAllWindows()


