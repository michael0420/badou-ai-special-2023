# 导入OpenCV库
import cv2
# 导入NumPy库
import numpy as np

# 通过OpenCV的imread函数读取图像文件
img = cv2.imread("photo1.jpg")
# 创建图像的副本，用于后续比较
img_copy = img.copy()
# 定义原始图像和目标图像上的对应点坐标数组
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
# 打印原始图像的形状（高度、宽度和通道数）
print(f"原始图像的形状(高度、宽度、通道数):\n{img.shape}")
# 使用cv2.getPerspectiveTransform计算透视变换矩阵
transform = cv2.getPerspectiveTransform(src, dst)
# 打印透视变换矩阵
print(f"透视变换矩阵:\n{transform}")
# 使用cv2.warpPerspective函数进行透视变换
result = cv2.warpPerspective(img_copy, transform, (337, 488))
# 显示原始图像和透视变换后的图像
cv2.imshow("Original image",img_copy)
cv2.imshow("Transform image",result)
# 等待用户按下任意键，然后关闭图像显示窗口
cv2.waitKey(0)
