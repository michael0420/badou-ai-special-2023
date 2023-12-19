# encoding=gbk

"""
@author: BraHitYQ
Canny edge detection(Canny边缘检测)——Canny函数用于边缘检测，它通过计算图像中每个像素点的梯度强度和方向来检测边缘。
The Canny function is used for edge detection, which detects edges by calculating the gradient intensity and direction of each pixel in the image.
"""


import cv2


'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
必要参数：
    image：输入图像，类型为cv2.typing.MatLike。该参数是需要处理的原图像，该图像必须为单通道的灰度图；
    threshold1：第一个阈值，类型为float。该参数是滞后阈值1；
    threshold2：第二个阈值，类型为float。该参数是滞后阈值2。
    edges：输出边缘图像，类型为cv2.typing.MatLike | None，默认值为None。
    apertureSize：高斯模糊的孔径大小，类型为int，默认值为3。
    L2gradient：是否使用L2范数计算梯度，类型为bool，默认值为False。
    
函数功能：
    函数返回一个类型为cv2.typing.MatLike的边缘图像。在函数内部，我们首先将输入图像转换为灰度图像（如果尚未转换），然后使用高斯模糊对图像进行平滑处理。接下来，我们调用cv2.Canny()函数来检测边缘，并将结果存储在detected_edges变量中。最后，我们返回检测到的边缘图像。
'''


img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 其实该步骤可以取消，因为canny函数自带图像灰度化
cv2.imshow("canny", cv2.Canny(gray, 200, 300))
cv2.waitKey()
cv2.destroyAllWindows()


"""
代码解析：
    1.导入cv2库
    2.使用cv2.imread()函数读取图片，参数1表示以彩色模式读取图片
    3.使用cv2.cvtColor()函数将彩色图像转换为灰度图像
    4.使用cv2.Canny()函数进行Canny边缘检测，参数200和300分别表示低阈值和高阈值
    5.使用cv2.imshow()函数显示处理后的图像
    6.使用cv2.waitKey()函数等待用户按键
    7.使用cv2.destroyAllWindows()函数关闭所有打开的窗口
"""