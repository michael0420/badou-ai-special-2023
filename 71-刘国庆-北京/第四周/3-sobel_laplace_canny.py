# 导入OpenCV库和Matplotlib库
import cv2
import matplotlib.pyplot as plt

# 读取彩色图像文件 "lenna.png"，存储在变量 img 中
img = cv2.imread("lenna.png")
# 将彩色图像转换为灰度图像，存储在变量 img_gray 中
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 使用Sobel算子进行水平方向边缘检测，结果存储在变量 img_sobel_x 中
img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
# 使用Sobel算子进行垂直方向边缘检测，结果存储在变量 img_sobel_y 中
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
# 使用Laplace算子进行边缘检测，结果存储在变量 img_laplace 中
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)
# 使用Canny算子进行边缘检测，结果存储在变量 img_canny 中
img_canny = cv2.Canny(img_gray, 100, 150)

# 使用Matplotlib库创建子图并显示图像
# 原始灰度图像
plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("Original")
# Sobel_x边缘检测结果
plt.subplot(232), plt.imshow(img_sobel_x, "gray"), plt.title("Sobel_x")
# Sobel_y边缘检测结果
plt.subplot(233), plt.imshow(img_sobel_y, "gray"), plt.title("Sobel_y")
# Laplace边缘检测结果
plt.subplot(234), plt.imshow(img_laplace, "gray"), plt.title("Laplace")
# Canny边缘检测结果
plt.subplot(235), plt.imshow(img_canny, "gray"), plt.title("Canny")
# 显示所有子图
plt.show()
