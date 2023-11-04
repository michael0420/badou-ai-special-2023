import cv2 
import numpy as np  
'''
功能：下采样 对源图像按比例缩放
''' 
def function(img): 
    # 获取图像的高度、宽度和通道数。
    height, width, channels = img.shape  
    # 创建一个空的图像，大小为800x800，通道数与输入图像相同，数据类型为无符号8位整数
    emptyImage = np.zeros((800, 800, channels), np.uint8) 
    sh = 800 / height  # 计算缩放比例，将图像高度缩放到800像素
    sw = 800 / width  # 计算缩放比例，将图像宽度缩放到800像素
    for i in range(800):  # 遍历新图像的行索引
        for j in range(800):  # 遍历新图像的列索引
            x = int(i / sh + 0.5)  # 根据缩放比例和当前位置计算在原图中对应的横坐标
            y = int(j / sw + 0.5)  # 根据缩放比例和当前位置计算在原图中对应的纵坐标
            emptyImage[i, j] = img[x, y]  # 将原图中对应位置的像素值赋给新图像中的对应位置
    return emptyImage  # 返回缩放后的图像

# cv2.resize(img, (800, 800, c), near/bin) 使用cv2.resize函数对图像进行缩放，但未提供具体参数

img = cv2.imread("lenna.png") 
zoom = function(img) 
print(zoom) 
print(zoom.shape)  # 打印缩放后图像的形状（尺寸）
cv2.imshow("nearest interp", zoom)  # 创建一个窗口显示缩放后的图像，窗口标题为"nearest interp"
cv2.imshow("image", img)  # 创建一个窗口显示原始图像，窗口标题为"image"
cv2.waitKey(0)  # 等待用户按下任意键关闭窗口
