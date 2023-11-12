import cv2
import numpy as np
img = cv2.imread('lenna.png',1)
cv2.imshow('img', img)
# 拆分BGR三通道
(b, g, r) = cv2.split(img)
bHist = cv2.equalizeHist(b)
gHist = cv2.equalizeHist(g)
rHist = cv2.equalizeHist(r)
res = cv2.merge((bHist, gHist, rHist))
cv2.imshow('rgb_Hist', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
