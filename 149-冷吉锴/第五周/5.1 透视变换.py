import cv2
import numpy as np

img = cv2.imread('photo1.jpg')
result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])  # 这张纸的4个顶点
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)
# 获得透视变换矩阵
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
# 进行透视变换
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
