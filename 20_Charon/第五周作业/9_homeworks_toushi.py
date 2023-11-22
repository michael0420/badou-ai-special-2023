import cv2
import numpy as np



img = cv2.imread('lenna2.jpg')

img_copy = img.copy()

src_points = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst_points = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

# 原始角点和目标角点的映射
trans = cv2.getPerspectiveTransform(src_points, dst_points)
dst_img = cv2.warpPerspective(img_copy, trans, (500, 500))

cv2.imshow('src_img', img)
cv2.imshow('dst_img', dst_img)
cv2.waitKey(0)

