# opencv-contrib-python==3.4.2.16
# opencv-python==3.4.2.16
# Python 3.6.13

import cv2
import numpy as np


def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))  # 连线

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


# img1_gray = cv2.imread("iphone1.png")
# img2_gray = cv2.imread("iphone2.png")
img1_gray = cv2.imread("lenna_rotate.jpg")
img2_gray = cv2.imread("lenna_sharp.jpg")

# sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()
# sift = cv2.SURF()

# kp关键点信息(位置,尺度,方向),des关键点描述符,128维的梯度信息
kp1, des1 = sift.detectAndCompute(img1_gray, None)  # img1_gray的关键点信息,
kp2, des2 = sift.detectAndCompute(img2_gray, None)  # img2_gray的关键点信息

# BFmatcher with default parms
bf = cv2.BFMatcher(cv2.NORM_L2)  # 二维特征点匹配,尝试所有可能的匹配，从而使得它总能够找到最佳匹配,穷举
matches = bf.knnMatch(des1, des2, k=2)  # knn邻近匹配

goodMatch = []#排除相似度不高的关键点
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()
