"""
SIFT——特征匹配
"""
import cv2
import numpy as np

"""得到好的特征匹配"""
def get_goodMatch(des1,des2):
    #创建特征匹配对象bf
    #默认使用L2范数进行距离计算
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    #进行特征匹配,k=2返回每个查询描述子的最佳匹配个数
    matches = bf.knnMatch(des1,des2,k=2)

    #筛选好的匹配结果，保存在goodMatch列表中
    goodMatch = []
    for m,n in matches:
        #如果第一个最佳匹配的距离小于第二个最佳匹配距离的0.5倍，则认为是好的匹配
        if m.distance < 0.5 * n.distance:
            goodMatch.append(m)

    return goodMatch


"""实现SIFT"""
#输入：两个图像+两个图像关键点+好的匹配结果
#绘制对应关键点匹配线
def drawMatchesKnn_cv2(img1_gray, img2_gray,kp1,kp2,goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    #创建空白图像，将两个图像链接在一起
    vis = np.zeros((max(h1,h2),w1+w2,3), np.uint8)
    vis[:h1,:w1] = img1_gray
    vis[:h2,w1:w1+w2] = img2_gray

    #从好的匹配结果中提取对应的关键点索引，p1为查询图像的，p2为训练图像的
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    #提取对应的关键点坐标
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1,0)

    #在vis图像上绘制匹配点之间连线
    for (x1,y1), (x2,y2) in zip(post1,post2):
        cv2.line(vis,(x1,y1),(x2,y2),(0,0,255))
    
    #创建窗口
    cv2.namedWindow('match',cv2.WINDOW_NORMAL)
    cv2.imshow('match',vis)

#读取图像
img1 = cv2.imread('iphone1.png')
img2 = cv2.imread('iphone2.png')

#创建SIFT对象
sift = cv2.xfeatures2d.SIFT_create()

#检测关键点并计算描述子,None表示没有掩码
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

#获取最佳匹配点
goodMatch = get_goodMatch(des1,des2)

#实现SIFT特征匹配
drawMatchesKnn_cv2(img1,img2,kp1,kp2,goodMatch)
cv2.waitKey(0)
cv2.destroyAllWindows()