import numpy as np
import cv2

img = cv2.imread('photo1.jpg')
tmp = img.copy()

src = np.float32([[206, 154], [518, 286], [14, 603], [343, 730]])
dst = np.float32([[0, 0], [480, 0], [0, 900], [480, 900]])

m = cv2.getPerspectiveTransform(src, dst)

result = cv2.warpPerspective(tmp, m, dsize=(480, 900))

cv2.imshow('result', result)
cv2.imshow('src', img)
cv2.waitKey(0)