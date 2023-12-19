import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread("photo1.jpg")  # 替换为你的图像路径

# 定义原始图像中四个角点的坐标
original_points = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])

# 定义目标图像中对应的四个角点的坐标（可以调整目标的形状和尺寸）
target_points = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

# 计算透视变换矩阵
perspective_matrix = cv2.getPerspectiveTransform(original_points, target_points)

# 进行透视变换
result_img = cv2.warpPerspective(img, perspective_matrix, (337, 488))

# 显示原始图像和透视变换后的图像
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title('Perspective Transformed Image')

plt.show()
