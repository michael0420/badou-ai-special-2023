import cv2
import numpy as np


def drawMatchesKnn_cv2(img1, kp1, img2, kp2, goodMatch):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
    vis = cv2.resize(vis,(2000,round(vis.shape[0]/vis.shape[1]*2000)))
    cv2.imshow("match", vis)


img1 = cv2.imread("pic.png")
img2 = cv2.imread("pic2.png")
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
img1 = cv2.drawKeypoints(image=img1, outImage=img1, keypoints=kp1,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                         color=(0, 0, 255))
img2 = cv2.drawKeypoints(image=img2, outImage=img2, keypoints=kp2,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                         color=(0, 0, 255))
cv2.imshow('kp1', img1)
cv2.imshow('kp2', img2)
cv2.waitKey(0)
matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
goodMatch = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)
drawMatchesKnn_cv2(img1, kp1, img2, kp2, goodMatch[:20])
cv2.waitKey(0)
