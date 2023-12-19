'''
读取图片
寻找顶点，指定要变换的顶点坐标
获取变换矩阵
透视变换操作
保存结果
'''
import cv2
import numpy as np

# 读取图片
img = cv2.imread('photo1.jpg')

# 指定顶点
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

# 生成透视变换矩阵
warpMatrix = cv2.getPerspectiveTransform(src, dst)
print(warpMatrix)

# 进行透视变换
res = cv2.warpPerspective(img, warpMatrix, (337, 488))

# 保存结果
cv2.imwrite('res.png', res)
