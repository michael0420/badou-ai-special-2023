import cv2
import numpy as np


def drawMatchesKnn_cv2(img1, kp1, img2, kp2, goodMatch):
    H1, W1, C1 = img1.shape
    H2, W2, C2 = img2.shape
    # 输出对比图，两张图横向连接，高度取最大，宽是和
    H_out, W_out = max(H1, H2), W1 + W2
    output_pic = np.zeros((H_out, W_out, 3), np.uint8)

    output_pic[:H1, :W1] = img1
    output_pic[:H2, W1:W1 + W2] = img2

    # 从匹配对中找到每张图中的匹配点索引
    img1_point_index = [kpp.queryIdx for kpp in goodMatch]
    img2_point_index = [kpp.trainIdx for kpp in goodMatch]
    # 根据索引找到点的坐标
    img1_point_xy = np.int32([kp1[pp].pt for pp in img1_point_index])
    img2_point_xy = np.int32([kp2[pp].pt for pp in img2_point_index]) + (W1, 0)
    # 根据每对匹配点的坐标画线
    for (x1, y1), (x2, y2) in zip(img1_point_xy, img2_point_xy):
        cv2.line(output_pic, (x1, y1), (x2, y2), (0, 0, 255))
 
    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", output_pic)


if __name__ == '__main__':
    # 1、读取两张图片
    img1 = cv2.imread("pic1.png")
    img2 = cv2.imread("pic2.png")

    # 2、分别找到两张图片特征点
    # 实例化sift模型
    sift = cv2.xfeatures2d.SIFT_create()
    # 调用模型分别找到两张图片的关键点
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3、特征点匹配
    # 实例化匹配模型
    bf = cv2.BFMatcher(cv2.NORM_L2)
    # knnMatch--K近邻匹配，在匹配的时候选择K个和特征点最相似的点，
    # 如果这K个点之间的区别足够大，则选择最相似的那个点作为匹配点，
    # 通常选择K = 2，也就是最近邻匹配。对每个匹配返回两个最近邻的匹配，
    # 如果第一匹配和第二匹配距离比率足够大（向量距离足够远），
    # 则认为这是一个正确的匹配，比率的阈值通常在2左右。
    matches = bf.knnMatch(des1, des2, k=2)

    goodMatch = []
    for m, n in matches:
        if m.distance < 0.50*n.distance:
            goodMatch.append(m)
    # 画出最匹配的50个点
    drawMatchesKnn_cv2(img1, kp1, img1, kp2, goodMatch[:50])

    cv2.waitKey(0)

