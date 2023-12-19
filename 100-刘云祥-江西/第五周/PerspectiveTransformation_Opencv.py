import cv2
import numpy as np
SrcImg = cv2.imread('photo1.jpg')
Result = SrcImg.copy()
InputPoint = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
OutputPoint = np.float32([[0, 0], [336, 0], [0, 487], [336, 487]])
print(type(OutputPoint))
M = cv2.getPerspectiveTransform(InputPoint, OutputPoint)  # 获取3*3变换矩阵
print(M)
DstImg = cv2.warpPerspective(Result, M, (337, 488))
cv2.imshow("SrcImg", SrcImg)
cv2.imshow("DstImg", DstImg)
cv2.imwrite("photo2.jpg", DstImg)
cv2.waitKey(0)
