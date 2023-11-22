import cv2
import numpy as np

img = cv2.imread('photo1.jpg')

img_copy = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 膨胀操作
edge = cv2.Canny(dilate, 30, 100, 4)  # 边缘检测
# cv2.imshow("edge", edge)
# cv2.waitKey(0)

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

print("docCnt", docCnt)

src = np.float32([tuple(point[0]) for point in docCnt])
dst = np.float32([[0, 0], [0, 488], [337, 488], [337, 0]])


# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(img_copy, m, (337, 488))
cv2.imshow("result", result)
cv2.waitKey(0)
