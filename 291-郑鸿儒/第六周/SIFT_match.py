#!/usr/bin/env python
# encoding=utf-8
import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_match_points(img_1, kp1, img_2, kp2, good_matches):
    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]

    # np.zeros 默认类型为float
    img_stack = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    img_stack[:h1, :w1] = img_1
    img_stack[:h2, w1: w1 + w2] = img_2

    # 取出所有match中的查询描述符索引和训练描述符索引
    pos_1 = [match.queryIdx for match in good_matches]
    pos_2 = [match.trainIdx for match in good_matches]

    # print(pos_1)
    # 关键点中含有方向，尺度，位置金字塔层数等信息，这里根据索引取出坐标
    start = np.int32([kp1[pos].pt for pos in pos_1])
    end = np.int32([kp2[pos].pt for pos in pos_2])
    # print(start)
    # 划线
    for (x1, y1), (x2, y2) in zip(start, end):
        # print(x1, y1, x2, y2)
        cv2.line(img_stack, (x1, y1), (x2 + w1, y2), (255, 0, 0))
    # 显式的创建窗口可以设置大小分辨率等参数，但不写的话matplotlib也会创建默认窗口
    plt.figure()
    cv2.imshow("match", img_stack)
    # 按下任意按键关闭(销毁)窗口
    if cv2.waitKey(0) != -1:
        cv2.destroyAllWindows()


img_1 = cv2.imread("img/iphone1.png")
img_2 = cv2.imread("img/iphone2.png")

# print(img_2)

# 创建sift对象
sift = cv2.SIFT_create()
# detectAndCompute 需要传入彩色图片
kp1, des1 = sift.detectAndCompute(img_1, None)
kp2, des2 = sift.detectAndCompute(img_2, None)

# BFMatch
# Brute-Force匹配器，适用于小规模的匹配任务
# Args
#   normaltype: 默认cv2.NORM_L2 表示欧氏距离
#   crosscheck: 布尔值，默认False，是否进行交叉验证
bf = cv2.BFMatcher()
# knnMatch
# 最近邻特征匹配，返回每个查询描述符的k个最佳匹配项、
# Args
#   queryDescriptors: 查询图像的特征描述符
#   trainDescriptors: 训练图像的特征描述符
#   k: 可选，表示每个查询描述符的最佳匹配项的数目
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    # 最佳匹配项欧式距离小于次级匹配项欧氏距离的一半则认为式相同特征点
    if m.distance < n.distance / 2:
        good_matches.append(m)

draw_match_points(img_1, kp1, img_2, kp2, good_matches)
