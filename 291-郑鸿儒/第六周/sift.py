import cv2
import numpy as np


def drawGoodMatch(img_1, kp1, img_2, kp2, goodMatches):
    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]

    # 合并图片
    # print(max(h1, h2), w1 + w2, 3)
    merge = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    merge[:h1, :w1] = img_1
    merge[:h2, w1: w1 + w2] = img_2

    p1 = [kpp.queryIdx for kpp in goodMatches]
    p2 = [kpp.trainIdx for kpp in goodMatches]

    print(kp1)
    # print(dir(kp1[1]))
    print(kp1[1].pt)
    print(kp1[1].angle)
    print(kp1[1].class_id)
    print(kp1[1].convert)
    print(kp1[1].octave)
    print(kp1[1].overlap)
    print(kp1[1].response)
    print(kp1[1].size)
    # print(kp2)

    # for kpp in goodMatches:
    #     print(1111, kpp.queryIdx)
    #     print(2222, kpp.trainIdx)

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + [w1, 0]

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(merge, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.namedWindow('match', cv2.WINDOW_NORMAL)
    cv2.imshow('match', merge)


img_1 = cv2.imread('iphone1.png')
img_2 = cv2.imread('iphone2.png')

sift = cv2.xfeatures2d.SIFT_create()

kp1, des_1 = sift.detectAndCompute(img_1, None)
kp2, des_2 = sift.detectAndCompute(img_2, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des_1, des_2, k=2)

goodMatches = []

for m, n in matches:
    if m.distance < 0.5 * n.distance:
        goodMatches.append(m)

drawGoodMatch(img_1, kp1, img_2, kp2, goodMatches)

cv2.waitKey()
cv2.destroyAllWindows()
