import cv2
import numpy as np
image = cv2.imread('photo1.jpg')

# 定义四个源点
src = np.float32([[206,150],[518,286],[16,602],[343,731]])
# 定义四个目标点
dst = np.float32([[0,0],[540,0],[0,960],[540,960]])

# 计算透视变换矩阵
m = cv2.getPerspectiveTransform(src,dst)

# 进行透视变换
result = cv2.warpPerspective(image,m,[540,960])

# 显示原图和变换后的图像
cv2.imshow('src_image:',image)
cv2.imshow('dst_image',result)
cv2.waitKey(0)
cv2.destroyAllWindows()