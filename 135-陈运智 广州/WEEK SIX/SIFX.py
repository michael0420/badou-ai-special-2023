# import cv2
# import numpy as np
#
#
# def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
#     h1, w1 = img1_gray.shape[:2]
#     h2, w2 = img2_gray.shape[:2]
#
#     vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
#     vis[:h1, :w1] = img1_gray
#     vis[:h2, w1:w1 + w2] = img2_gray
#
#     p1 = [kpp.queryIdx for kpp in goodMatch]
#     p2 = [kpp.trainIdx for kpp in goodMatch]
#
#     post1 = np.int32([kp1[pp].pt for pp in p1])
#     post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
#
#     for (x1, y1), (x2, y2) in zip(post1, post2):
#         cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
#
#     cv2.namedWindow("match", cv2.WINDOW_NORMAL)
#     cv2.imshow("match", vis)
#
#
# img1_gray = cv2.imread("iphone1.png")
# img2_gray = cv2.imread("iphone2.png")
#
# # sift = cv2.SIFT()
# sift = cv2.SIFT_create()
# # sift = cv2.SURF()
#
# kp1, des1 = sift.detectAndCompute(img1_gray, None)
# kp2, des2 = sift.detectAndCompute(img2_gray, None)
#
# # BFmatcher with default parms
# bf = cv2.BFMatcher(cv2.NORM_L2)
# matches = bf.knnMatch(des1, des2, k=2)
#
# goodMatch = []
# for m, n in matches:
#     if m.distance < 0.50 * n.distance:
#         goodMatch.append(m)
#
# drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()



import cv2
import numpy as np

# 读取两张图片
img1 = cv2.imread("human1.jpg")
img2 = cv2.imread("human2.jpg")

# 使用 SIFT 算法创建 SIFT 对象
sift = cv2.SIFT_create()

# 检测关键点并计算描述符
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# 使用 Brute-Force 匹配器
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 应用比例测试，保留良好的匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 绘制匹配结果
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 显示结果
cv2.namedWindow("SIFT Matches", cv2.WINDOW_NORMAL)
cv2.imshow("SIFT Matches", img_matches)

# cv2.imshow("SIFT Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()