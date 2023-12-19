import cv2
import numpy as np


# 读取原始图
img = cv2.imread("photo1.jpg")
# 创建图像的副本
img_copy = img.copy()
# 定义原始图像和目标图像上的对应点坐标数组
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [500, 0], [0, 700], [500, 700]])
# 计算透视变换矩阵
transform = cv2.getPerspectiveTransform(src, dst)
# 透视变换
result = cv2.warpPerspective(img_copy, transform, (500, 700))
# 显示原始图像和透视变换后的图像
cv2.imshow("Original image",img_copy)
cv2.imshow("Transform image",result)
cv2.waitKey(0)