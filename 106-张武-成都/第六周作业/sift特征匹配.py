import cv2
import numpy as np


# 读图
gray1 = cv2.imread('iphone1.png',cv2.IMREAD_GRAYSCALE)
gray2 = cv2.imread('iphone2.png',cv2.IMREAD_GRAYSCALE)
# 初始化sift
sift = cv2.xfeatures2d.SIFT_create()

# 获取关键点和特征
kp1,des1 = sift.detectAndCompute(gray1,None)
kp2,des2 = sift.detectAndCompute(gray2,None)

# 初始化BF匹配器
bf = cv2.BFMatcher(cv.NORM_L2)
# 获得k个最佳匹配
matches = bf.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append(m)

img = cv2.drawMatchesKnn(gray1,kp1,gray2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('img',img)