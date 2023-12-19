#!/usr/bin/env python
# encoding=utf-8
import cv2
import numpy


gray = cv2.imread("img/lenna.png", 0)
cv2.imshow("Canny", cv2.Canny(gray, 100, 200))
cv2.waitKey(0)
