import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库


# 定义一个函数drawMatchesKnn_cv2，用于绘制两张图像之间匹配的特征点
# image_1: 第一张图像。
# kp1: 第一张图像的关键点。
# image_2: 第二张图像。
# kp2: 第二张图像的关键点。
# goodMatch: 一个包含匹配点信息的列表，这些匹配点是在两张图像之间经过筛选的良好匹配。
def drawMatchesKnn_cv2(img1, kp1, img2, kp2, goodMatch):
    # 获取图像1和图像2的高度和宽度
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # 创建一个新的图像，宽度为两个图像之和，高度为最大高度
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    # 将图像1复制到新图像的左侧
    vis[:h1, :w1] = img1
    # 将图像2复制到新图像的右侧
    vis[:h2, w1:w1 + w2] = img2
    # 提取goodMatch中每个匹配点在第一张图像中的特征点索引
    p1 = [kpp.queryIdx for kpp in goodMatch]
    # 提取goodMatch中每个匹配点在第二张图像中的特征点索引
    p2 = [kpp.trainIdx for kpp in goodMatch]
    # 获取goodMatch中每个匹配点在第一张图像中对应的特征点的坐标
    post1 = np.int32([kp1[pp].pt for pp in p1])
    # 获取goodMatch中每个匹配点在第二张图像中对应的特征点的坐标，注意对坐标进行水平偏移(w1, 0)
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
    # 在新图像上绘制连接特征点的直线
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), color=(255, 255, 255))
    # 创建一个显示窗口，显示绘制的匹配结果
    cv2.namedWindow("matches", cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("matches", vis)


# 读取第一张图像
img1 = cv2.imread("iphone1.png")
# 读取第二张图像
img2 = cv2.imread("iphone2.png")
# 创建SIFT对象
sift = cv2.xfeatures2d_SIFT.create()
# 使用SIFT算法检测并计算图像1的关键点和描述符
kp1, des1 = sift.detectAndCompute(img1, None)
# 使用SIFT算法检测并计算图像2的关键点和描述符
kp2, des2 = sift.detectAndCompute(img2, None)
# 创建Brute-Force匹配器对象，使用欧氏距离作为相似度度量
bf = cv2.BFMatcher(cv2.NORM_L2)
# 对两组描述符进行最近邻匹配，k=2表示每个描述符的两个最近邻
matches = bf.knnMatch(des1, des2, k=2)
# 筛选出距离比例小于0.5的匹配，将其添加到Match列表中
Match = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        Match.append(m)
# 调用绘制函数，显示前20个好的匹配
drawMatchesKnn_cv2(img1, kp1, img2, kp2, Match)
# 等待用户按下任意键
cv2.waitKey(0)
# 关闭显示窗口
cv2.destroyAllWindows()
