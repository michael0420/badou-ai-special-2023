import cv2
import numpy as np

# 读取图像
img = cv2.imread('lenna.png',0)

# 归一化方式3：手动计算归一化
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)
normalized_img3 = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# 显示归一化后的图像
cv2.imshow('Normalized Image 3', normalized_img3)
cv2.waitKey(0)
cv2.destroyAllWindows()