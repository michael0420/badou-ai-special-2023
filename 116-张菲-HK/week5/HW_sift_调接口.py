import cv2
import numpy as np

def drawMatchesKnn_cv2(img1, img2, kp1, kp2, goodmatch):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 创建一个空的黑的图片
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    # 把两个图片左右放在一起,img1在左边，img2在右边
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2

    p1 = [kpp.queryIdx for kpp in goodmatch] # kpp代表good Match列表中每个“DMatch”对象。是个变量名字
    p2 = [kpp.trainIdx for kpp in goodmatch]

    post1 = np.int32([kp1[pp].pt for pp in p1]) # pt是关键点的坐标,从 kp1 和 kp2 中分别获取匹配特征点的坐标
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0) # 加上偏移量，将 kp2 的特征点坐标放在右边

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL) # 有cv2.WINDOW_NORMAL参数时，可以拉动调整窗口
    cv2.imshow("match", vis)

img1 = cv2.imread("iphone1.png")
img2 = cv2.imread("iphone2.png")

sift = cv2.SIFT_create()

# 第二个参数是掩码，如果想限制在图像的某个区域内检测关键点，可以提供一个掩码图像，None 是在整个图像上进行检测。
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2) # 创建了一个 BFMatcher 对象，使用欧氏距离（L2范数）作为匹配度量, NORM_HANMING是汉明距离
matches = bf.knnMatch(des1, des2, k=2) # 使用 knnMatch 来获取两幅图像的匹配结果

goodMatch = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

drawMatchesKnn_cv2(img1, img2, kp1, kp2, goodMatch[:20])

cv2.waitKey(0)




