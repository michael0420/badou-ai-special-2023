import cv2

# 读取图像
img = cv2.imread('lenna.png',1)

# print("OpenCV version:", cv2.__version__)
# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 检测关键点和计算描述符
keypoints, descriptors = sift.detectAndCompute(img, None)

# 在图像上绘制关键点,cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向
img_with_keypoints = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                                       flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
                                       color=(205,0,0))

# 显示结果
cv2.imshow('Image with Keypoints', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()