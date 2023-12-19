import cv2
import numpy as np

def kmeans(image, k):
    # 将图像转换为一维数组
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    # 定义K-means的相关参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    # 应用K-means算法
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)
    
    # 将每个像素点的标签转换为对应的中心点值
    segmented_image = centers[labels.flatten()].reshape(image.shape)
    
    return segmented_image

# 读取图像
image = cv2.imread('image.jpg')

# 对图像进行分割，得到4个簇
k = 4
segmented_image = kmeans(image, k)

# 显示原始图像和分割后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()