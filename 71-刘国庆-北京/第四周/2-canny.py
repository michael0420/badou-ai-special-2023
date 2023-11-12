import cv2
import numpy as np

# 读取彩色图像，并存储在img变量中
img = cv2.imread("lenna.png")
# 将彩色图像转换为灰度图像
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 使用Canny边缘检测算法处理灰度图像
# 参数200和300分别为低阈值和高阈值，控制边缘检测的灵敏度
img_canny = cv2.Canny(img_gray, 200, 300)
# 在同一窗口中显示原始灰度图像和使用Canny处理后的图像，np.hstack用于将两幅图像水平堆叠,窗口标题为"Gray->Canny"
cv2.imshow("Gray->Canny", np.hstack([img_gray, img_canny]))
# 等待用户按下任意键
cv2.waitKey()
# 销毁所有打开的窗口，释放内存
cv2.destroyAllWindows()
