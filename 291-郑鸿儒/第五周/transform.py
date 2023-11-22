import cv2
import numpy as np

img = cv2.imread('phone.jpg')

img_copy = img.copy()
src = np.float32([[120, 140], [314, 52], [400, 461], [610, 335]])
dst = np.float32([[0, 0], [240, 0], [0, 435], [240, 435]])
matrix = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(img_copy, matrix, (240, 435))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
