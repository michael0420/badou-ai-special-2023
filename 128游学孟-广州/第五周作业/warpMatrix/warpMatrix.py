import cv2
import numpy as np

#目标平面尺寸
target_width = 200
target_height = 200


# 源坐标点，这里选取图像中的一个矩形区域
source_points = np.array([[100,100],[700,100],[100,400],[700,400]],dtype=np.float32)

# 目标坐标点
target_points = np.array([[target_height,0],[target_height,target_width],[0,0],[0,target_height]],dtype=np.float32)

# 计算透视变换矩阵
perspective_matrix = cv2.getPerspectiveTransform(source_points,target_points)

# 读取输入图像
image = cv2.imread('D://xuexi//zuoye//week5//warpMatrix//warpMatrix.jpeg')

# 进行透视变换
projected_image = cv2.warpPerspective(image,perspective_matrix,(target_width,target_height))

# 显示原始图像
cv2.imshow('Original Image',image)

# 显示投影后的图像
cv2.imshow('Projected Image',projected_image)

# 等待键盘输入
cv2.waitKey(0)

