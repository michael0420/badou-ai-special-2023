# 导入OpenCV库
import cv2

# 使用OpenCV的imread函数从文件中读取名为 "lenna.png" 的图像，并将其存储在变量img中
img = cv2.imread("lenna.png")
# 将彩色图像转换为灰度图像
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建一个SIFT检测器对象
sift = cv2.xfeatures2d_SIFT.create()
# 使用SIFT检测器在灰度图像上检测关键点并计算关键点的描述符
keypoints, descritor = sift.detectAndCompute(img_gray, None)
# 在图像上绘制关键点，包括圆圈和方向
sift_keypoints = cv2.drawKeypoints(image=img,
                                   outImage=img,
                                   keypoints=keypoints,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 0, 0))
# 使用OpenCV的imshow函数显示包含SIFT关键点的图像，窗口标题为 "sift_keypoints"
cv2.imshow("sift_keypoints", sift_keypoints)
# 等待用户按下任意键，参数0表示无限等待
cv2.waitKey(0)
# 关闭所有OpenCV的图像窗口
cv2.destroyAllWindows()
