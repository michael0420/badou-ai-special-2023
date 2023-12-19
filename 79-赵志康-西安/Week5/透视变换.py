import cv2
import numpy as np
img = cv2.imread("1.jpeg")
'''
注意这里pst1和pst2的输入并不是图像，而是图像对应的顶点坐标。
'''
pts1 = np.float32([[593, 237], [1897, 478], [60, 925], [1732, 1442]])#
pts2 = np.float32([[0, 0], [1200, 0], [0, 600], [1200, 600]])
"""
cv2.getPerspectiveTransform(src, dst)是OpenCV库中的一个函数，用于计算透视变换矩阵。
透视变换是一种二维图像变换技术，它可以将一个平面上的点映射到另一个平面上，同时保持形状和大小不变。
这个函数需要两个参数：src和dst，分别表示原始图像中的四个顶点坐标和目标图像中的四个顶点坐标。
函数返回一个3x3的透视变换矩阵。
"""
M = cv2.getPerspectiveTransform(pts1, pts2)
print("wrapMatrix:")
print(M)
"""
使用cv2.warpPerspective()函数将源图像进行透视变换，并将结果保存在result变量中
cv2.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])
src：输入图像，通常为灰度图像或彩色图像。
M：3x3的透视变换矩阵。
dsize：输出图像的大小，以像素为单位。
dst：输出图像，可选参数，默认为None。
flags：插值方法，可选参数，默认为INTER_LINEAR。
borderMode：边界填充模式，可选参数，默认为BORDER_CONSTANT。
borderValue：边界填充值，可选参数，默认为0。
"""
result = cv2.warpPerspective(img, M, (1200, 600))
cv2.imshow("Source", img)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
