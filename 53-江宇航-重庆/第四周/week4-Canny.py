import cv2
import numpy as np

# 读取Lenna图，并将其灰度化
img = cv2.imread("C:/Users/15082/Desktop/lenna.png")
greyimg =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 减少图像中的高斯噪声
Gussianimg = cv2.GaussianBlur(greyimg, (3, 3), 0)

# 分别打印消没有消除高斯噪声的灰度图和消除了高斯噪声的灰度图
cv2.imshow('CannyGrey', cv2.Canny(greyimg, 100, 200))
cv2.imshow('CannyGussian', cv2.Canny(Gussianimg, 100, 200))

cv2.waitKey(0)
cv2.destroyAllWindows()