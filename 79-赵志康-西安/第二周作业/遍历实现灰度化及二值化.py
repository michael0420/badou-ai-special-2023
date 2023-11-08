import numpy as np
import cv2

img = cv2.imread("123.png")            #读取图片
if img.any() == None:                        #防呆判断
    print('fail to load image!')
h,w = img.shape[:2]                    #获取图片的长宽
"""
img.shape[:2] 取彩色图片的长、宽。
如果img.shape[:3] 则取彩色图片的长、宽、通道。
关于img.shape[0]、[1]、[2]
img.shape[0]：图像的垂直尺寸（高度）
img.shape[1]：图像的水平尺寸（宽度）
img.shape[2]：图像的通道数
"""
img_gray = np.zeros((h,w),img.dtype)
#创建一张和当前图片大小一样的单通道图片。img.dtype函数
# uint8表示无符号整数，没有符号位，8个比特位全部用来表示整数，所以数据范围是0到255。
for i in range(h):
    for j in range(w):
        m = img[i,j]                    #取出当前h和w中的BGR坐标
        img_gray[i,j]=int(m[0]*0.11 +m[1]*0.59+m[2]*0.3) #将BGR坐标转化成gray坐标并赋值给新图像
print(img_gray)
print("image show gray :%s"%img_gray)
cv2.imshow("image show ",img)
cv2.imshow("image show gray",img_gray)
cv2.waitKey()