import cv2
import numpy as np

img = cv2.imread('photo1.jpg')

input = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
output = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

# 得到变换矩阵
change_Matrix = cv2.getPerspectiveTransform(input, output)
# 应用变换矩阵得到结果
img_output = cv2.warpPerspective(img, change_Matrix, (337, 488))



cv2.imshow("img_input", img)
cv2.imshow("img_output", img_output)
cv2.waitKey(0)
