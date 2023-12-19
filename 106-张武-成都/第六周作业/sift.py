import cv2


# 1.读图
img = cv2.imread('lenna.png')
# 2.转灰度图
gary = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 创建sift对象
sift = cv2.xfeatures2d.SIFT_create()
# 获取关键点和特征描述
kp,des = sift.detectAndCompute(gray,None)
print(kp,des)
# 画圆和方向信息
img = cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('img kp',img)
cv2.waitKey(0)
cv2.destroyAllWindows()