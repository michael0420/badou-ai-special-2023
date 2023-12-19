import cv2
import numpy as np


img = cv2.imread('photo1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))#膨胀
edged = cv2.Canny(dilate, 30, 120, 3)            # 边缘检测

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
cnts = cnts[0]
#if imutils.is_cv2() else cnts[1]  # 判断是opencv2还是opencv3
docCnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # 根据轮廓面积从大到小排序
    for c in cnts:
        peri = cv2.arcLength(c, True)                           # 计算轮廓周长
        approx = cv2.approxPolyDP(c, 0.02*peri, True)           # 轮廓多边形拟合
        # 轮廓为4个点表示找到纸张
        if len(approx) == 4:
            docCnt = approx
            break

src = np.zeros(shape=(4,2))
dst = np.zeros(shape=(4,2))
i=0
center_x=0;
center_y=0;
for peak in docCnt:
    peak = peak[0]
    src[i]=peak
    i+=1
    center_x+=peak[0]
    center_y += peak[1]
center_x/=4
center_y/=4
src=src.astype(np.float32)
dst = np.float32([[0, 0], [0, 588], [437, 588], [437, 0]])



result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
# src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
# dst = np.float32([[0, 0], [437, 0], [0, 588], [437, 588]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (437, 588))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
