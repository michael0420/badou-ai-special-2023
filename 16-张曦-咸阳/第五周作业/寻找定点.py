import cv2

# import imutils

'''
cv2.approxPolyDP() 多边形逼近
作用:
对目标图像进行近似多边形拟合，使用一个较少顶点的多边形去拟合一个曲线轮廓，要求拟合曲线与实际轮廓曲线的距离小于某一阀值。

函数原形：
cv2.approxPolyDP(curve, epsilon, closed) -> approxCurve

参数：
curve ： 图像轮廓点集，一般由轮廓检测得到
epsilon ： 原始曲线与近似曲线的最大距离，参数越小，两直线越接近
closed ： 得到的近似曲线是否封闭，一般为True

返回值：
approxCurve ：返回的拟合后的多边形顶点集。
'''

img = cv2.imread('photo1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 膨胀操作
edge = cv2.Canny(dilate, 30, 100, 4)  # 边缘检测
cv2.imshow("edge", edge)
cv2.waitKey(0)

# 轮廓检测
cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0]

docCnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 根据轮廓面积从大到小排序
    for c in cnts:
        peri = cv2.arcLength(c, True)  # 计算轮廓周长
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 轮廓多边形拟合
        # 轮廓为4个点表示找到纸张
        if len(approx) == 4:
            docCnt = approx
            break

for peak in docCnt:
    peak = peak[0]
    print( "peak = " , peak)
    cv2.circle(img, tuple(peak), 10, (255, 0, 0))

cv2.imshow('img', img)
cv2.waitKey(0)
