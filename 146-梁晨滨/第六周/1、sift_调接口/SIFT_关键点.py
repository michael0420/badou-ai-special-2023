import cv2
import numpy as np

# 1、读取图像转成灰度
img = cv2.imread("pic1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2、实例化sift算法
sift = cv2.xfeatures2d.SIFT_create()
# 调用sift的detectAndCompute提取特征点，
# keypoints包含特征点的所有信息，descriptor是
keypoints, descriptor = sift.detectAndCompute(gray, None)

# 3、展示图片
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向。
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))

cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)

