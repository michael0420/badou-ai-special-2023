#!/usr/bin/env python
# encoding=utf-8
import cv2
import numpy as np


img = cv2.imread("./img/photo1.jpg")

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
m = cv2.getPerspectiveTransform(src, dst)
res = cv2.warpPerspective(img, m, (337, 488))
cv2.imshow("new image", res)
cv2.waitKey(0)
