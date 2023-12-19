import cv2
import numpy as np


img = cv2.imread('newphoto.jpg')
src = np.float32([[359, 425], [829, 514], [460, 1087], [1169, 1074]])
dst = np.float32([[0, 0], [600, 0], [0, 800], [600, 800]])
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:",m)
result = cv2.warpPerspective(img, m, (600, 800))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
