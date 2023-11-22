import cv2
import numpy as np

img = cv2.imread('photo1.jpg')
result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[330, 550], [660, 440], [110, 770], [220, 880]])
dst = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (400, 600))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
