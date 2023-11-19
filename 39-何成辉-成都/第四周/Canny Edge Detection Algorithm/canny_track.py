# encoding=gbk


"""
@author: BraHitYQ
Canny edge detection: optimized program——Adjustable threshold range(Canny边缘检测：优化的程序——可调节阈值范围),
"""


import cv2


def CannyThreshold(lowThreshold):  
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯滤波,这一行使用OpenCV库中的cv2.GaussianBlur函数对灰度图像进行高斯模糊处理。gray是输入的灰度图像，(3, 3)表示高斯核的大小为3x3，0表示标准差为0。处理后的结果存储在变量detected_edges中。
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold*ratio, apertureSize=kernel_size)  # 边缘检测,这一行使用OpenCV库中的cv2.Canny函数对经过高斯模糊处理后的图像进行边缘检测。detected_edges是输入的图像，lowThreshold和lowThreshold*ratio分别表示低阈值和高阈值，apertureSize表示Sobel算子的孔径大小。处理后的结果仍然存储在变量detected_edges中。

    # just add some colours to edges from original image.
    dst = cv2.bitwise_and(img, img, mask=detected_edges)  # 用原始颜色添加到检测的边缘上,这一行使用OpenCV库中的cv2.bitwise_and函数将原始图像与边缘检测结果进行按位与操作。img是原始图像，mask=detected_edges表示使用边缘检测结果作为掩码。处理后的结果存储在变量dst中
    cv2.imshow('canny demo', dst)  # 这一行使用OpenCV库中的cv2.imshow函数显示处理后的图像。窗口标题为'canny demo'，显示的内容为变量dst。


#  全局变量申明


lowThreshold = 0  
max_lowThreshold = 100  
ratio = 3  
kernel_size = 3  
  
img = cv2.imread('lenna.png')  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换彩色图像为灰度图
  
cv2.namedWindow('canny demo')  
  
# 设置调节杠,
'''
下面是第二个函数，cv2.createTrackbar()
共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名
'''

# 这行代码创建了一个滑动条，用于调整Canny边缘检测算法中的最小阈值。滑动条的初始值由变量lowThreshold指定，最大值由变量max_lowThreshold指定。当滑动条的值改变时，会调用函数CannyThreshold来更新阈值。
cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)
# 这行代码初始化了Canny边缘检测算法的最小阈值为0。这是通过调用前面定义的CannyThreshold函数实现的。
CannyThreshold(0)  # initialization  
if cv2.waitKey(0) == 27:  # wait for ESC key to exit cv2,这行代码等待用户按下ESC键。如果用户按下了ESC键（ASCII码为27），则执行下一行代码。
    cv2.destroyAllWindows()  # 这行代码关闭所有打开的窗口。在这段代码中，它关闭了显示Canny边缘检测结果的窗口。
