'''
1. 生成高斯差分金字塔（DOG金字塔），尺度空间构建
2. 空间极值点检测（关键点的初步查探
3. 稳定关键点的精确定位
4. 稳定关键点方向信息分配
5。 关键点描述
6. 特征点匹配
'''

import cv2
import numpy as np

img = cv2.imread("iphone1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray, None)

img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))

cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


