#!/usr/bin/env python
# encoding=utf-8
import cv2


def CannyThreshold(low_threshold):
    img_gauss = cv2.GaussianBlur(img_gray, (5, 5), 0)
    detection_edges = cv2.Canny(img_gauss, low_threshold, low_threshold * ratio, apertureSize=5)
    dst = cv2.bitwise_and(img, img, mask=detection_edges)
    # 需要与namedwindow 相同，否则会分成两个窗口
    cv2.imshow("Canny Demo", dst)


img = cv2.imread("img/lenna.png", 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
low_thureshold = 0
max_threshold = 300
ratio = 3

cv2.namedWindow('Canny Demo')

cv2.createTrackbar("min_threshold", "Canny Demo", low_thureshold, max_threshold, CannyThreshold)

CannyThreshold(0)
cv2.waitKey(0)
