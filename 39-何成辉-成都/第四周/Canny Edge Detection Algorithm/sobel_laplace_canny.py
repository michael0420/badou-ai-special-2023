# encoding=gbk

"""
@author: BraHitYQ
1.Edge detection in the x and y directions of the soft operator。（soble算子x，y方向的边缘检测）
2.Laplacian operator edge detection。（laplacian算子边缘检测）
3.Canny operator edge detection。（canny算子边缘检测）
"""


import cv2
from matplotlib import pyplot as plt  

img = cv2.imread("lenna.png", 1)

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  

'''
Sobel算子
Sobel算子函数原型如下：
dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) 
前四个是必须的参数：
第一个参数是需要处理的图像；
第二个参数是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。
其后是可选的参数：
dst是目标图像；
ksize是Sobel算子的大小，必须为1、3、5、7。
scale是缩放导数的比例常数，默认情况下没有伸缩系数；
delta是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
borderType是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
'''


# 如下两行代码使用Sobel算子计算图像在x方向上的梯度。cv2.Sobel()函数的参数解释如下：
# img_gray：输入的灰度图像。
# cv2.CV_64F：输出图像的数据类型，这里是64位浮点数。
# 1：表示在x方向上计算梯度。
# 0：表示在y方向上不计算梯度。
# ksize=3：Sobel算子的核大小，这里设置为3x3。

img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # 对x求导
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # 对y求导

# Laplace 算子,这行代码使用拉普拉斯算子计算图像的二阶导数。与前两行类似：
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)  

# Canny 算子
# img_gray: 这是一个灰度图像，表示输入的原始图像。
# 100: 这是Canny边缘检测算法的两个阈值参数之一，用于确定边缘的强度。较小的值将导致更宽松的边缘检测，较大的值将导致更严格的边缘检测。
# 150: 这是Canny边缘检测算法的另一个阈值参数之一，用于确定边缘的强度。较小的值将导致更宽松的边缘检测，较大的值将导致更严格的边缘检测。
img_canny = cv2.Canny(img_gray, 100 , 150)  

plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("Original")  
plt.subplot(232), plt.imshow(img_sobel_x, "gray"), plt.title("Sobel_x")  
plt.subplot(233), plt.imshow(img_sobel_y, "gray"), plt.title("Sobel_y")  
plt.subplot(234), plt.imshow(img_laplace,  "gray"), plt.title("Laplace")  
plt.subplot(235), plt.imshow(img_canny, "gray"), plt.title("Canny")  
plt.show()  
