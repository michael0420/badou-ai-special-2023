import cv2
import numpy as np

# 读取待处理的图像
image = cv2.imread('lenna.png')

# 定义透视变换的四个顶点坐标
src_points = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
dst_points = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])

# 计算透视变换矩阵
perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# 进行透视变换
output_image = cv2.warpPerspective(image, perspective_matrix, (200, 200))

# 显示结果
cv2.imshow('Input', image)

cv2.imshow('Output', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()