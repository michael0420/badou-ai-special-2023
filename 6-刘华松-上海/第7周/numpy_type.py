import cv2
import numpy as np

# 读取图像
img = cv2.imread('lenna.png',0)

# 归一化方式2：使用Numpy进行归一化
normalized_img2 = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# 将归一化后的图像转换为8位无符号整型
normalized_img2 = (255*normalized_img2).astype(np.uint8)

# 显示归一化后的图像
cv2.imshow('Normalized Image 2', normalized_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()