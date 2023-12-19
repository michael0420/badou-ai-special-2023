import cv2

# 读取图像
image = cv2.imread('lenna.png')

# 创建SIFT对象
sift = cv2.SIFT_create()

# 检测关键点
keypoints, descriptors = sift.detectAndCompute(image, None)

# 绘制关键点
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# 显示图像
cv2.imshow('Image with Keypoints', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()