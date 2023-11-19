import cv2
import numpy as np
'''
Canny算法的实现步骤
1、对图像进行灰度化
2、对图像进行高斯滤波：根据像素点及其领域点的灰度值按照一定的参数规则进行加权平均。
3、检测图像中的水平、垂直和对角边缘（如Prewit,Sobel算子）。
4、对梯度幅值进行非极大值抑制。
5、用双领域算法检测和连接边缘。
'''
# 读取图像
image = cv2.imread('lenna.png')

# 1. 灰度化
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 高斯滤波
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 3. 计算图像梯度
dx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
dy = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

# 计算梯度幅值和方向
gradient_magnitude = np.sqrt(dx**2 + dy**2)
gradient_direction = np.arctan2(dy, dx) * (180 / np.pi)

# 4. 非极大值抑制
edges = np.zeros_like(gradient_magnitude, dtype=np.uint8)

for i in range(1, gradient_magnitude.shape[0] - 1):
    for j in range(1, gradient_magnitude.shape[1] - 1):
        angle = gradient_direction[i, j]

        # 将角度调整为0到180度范围
        if angle < 0:
            angle += 180

        # 根据梯度方向进行非极大值抑制
        if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
            if (gradient_magnitude[i, j] > gradient_magnitude[i, j - 1]) and (gradient_magnitude[i, j] > gradient_magnitude[i, j + 1]):
                edges[i, j] = gradient_magnitude[i, j]
        elif (22.5 <= angle < 67.5):
            if (gradient_magnitude[i, j] > gradient_magnitude[i - 1, j + 1]) and (gradient_magnitude[i, j] > gradient_magnitude[i + 1, j - 1]):
                edges[i, j] = gradient_magnitude[i, j]
        elif (67.5 <= angle < 112.5):
            if (gradient_magnitude[i, j] > gradient_magnitude[i - 1, j]) and (gradient_magnitude[i, j] > gradient_magnitude[i + 1, j]):
                edges[i, j] = gradient_magnitude[i, j]
        elif (112.5 <= angle < 157.5):
            if (gradient_magnitude[i, j] > gradient_magnitude[i - 1, j - 1]) and (gradient_magnitude[i, j] > gradient_magnitude[i + 1, j + 1]):
                edges[i, j] = gradient_magnitude[i, j]

# 5. 双阈值算法检测和连接边缘
low_threshold = 30
high_threshold = 100

strong_edges = (edges > high_threshold).astype(np.uint8) * 255
weak_edges = ((edges >= low_threshold) & (edges <= high_threshold)).astype(np.uint8) * 255

# 连接弱边缘
connected_edges = cv2.connectedComponentsWithStats(weak_edges, connectivity=8)
labels = connected_edges[1]
for label in range(1, connected_edges[0]):
    points = np.column_stack(np.where(labels == label))
    if points.shape[0] > 0:
        strong_edges[points[:, 0], points[:, 1]] = 255

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Canny Edges', strong_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


