#!/usr/bin/env python
# encoding=utf-8
import cv2


img = cv2.imread("./img/lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
# sift = cv2.xfeatures2D.SIFT_create() # 旧版本中使用，新版opencv已经将其移除
keypoints, descriptions = sift.detectAndCompute(gray, None)

print(keypoints, descriptions)
img = cv2.drawKeypoints(img, keypoints, img, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("keypoints", img)
cv2.waitKey(0)
