import cv2

img = cv2.imread("photo.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 高斯滤波
img_Gaussian = cv2.GaussianBlur(img_gray, (5, 5), 0)
# 膨胀
img_dilate = cv2.dilate(img_Gaussian, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
# 边缘检测
img_canny = cv2.Canny(img_dilate, 30, 120, apertureSize = 3)
# 轮廓检测
contours, hierarchy = cv2.findContours(img_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

vertex = None
if len(contours) > 0:
    # 将轮廓的面积从大到小排列
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        # 计算轮廓的周长
        peri = cv2.arcLength(c, True)
        # 轮廓多边形拟合
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        # 轮廓为4个点表示找到纸张
        if len(approx) == 4:
            vertex = approx
            break

for peak in vertex:
    peak = peak[0]
    cv2.circle(img, tuple(peak), 10, (0, 0, 0), 5)

cv2.imshow("result", img)
cv2.waitKey()