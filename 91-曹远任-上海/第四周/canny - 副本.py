#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np


def func(x):
    global low, high
    low = cv2.getTrackbarPos('low threshold', 'canny')
    high = cv2.getTrackbarPos('high threshold', 'canny')
    cv2.imshow("canny", cv2.Canny(gray, low, high))


img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray)
cv2.namedWindow('canny')
low = 0
high = 0

cv2.createTrackbar('low threshold', 'canny', low, 1000, func)
cv2.createTrackbar('high threshold', 'canny', high, 1000, func)
func(0)
cv2.waitKey()
cv2.destroyAllWindows()
