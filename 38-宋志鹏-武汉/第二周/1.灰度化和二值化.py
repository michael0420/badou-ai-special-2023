from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 使用plt.subplot创建小图，221表示创建2行2列，当前位置是1
plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna---")
print(img)


# # 灰度化
# img = cv2.imread("lenna.png")
# # 获取图片的high和wide
# h,w = img.shape[:2]
# # 创建一张和当前图片大小一样的单通道图片
# img_gray = np.zeros([h,w],img.dtype)
# for i in range(h):
#     for j in range(w):
#         # 取出当前high和wide的BGR坐标
#         m = img[i,j]
#         # 将BGR坐标转换为gray坐标并赋值给新图像
#         # 浮点算法：Gray = R0.3 + G0.59 + B0.11
#         img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
# print (img_gray)
# print ("image show gray:%s"%img_gray)
# cv2.imshow("image show gray",img_gray)
# # cv2.imshow('image',img)
# # 等待键盘输入，0表示无限等待，直到按下任意键
# # cv2.waitKey(0)




# 灰度化
img_gray = rgb2gray(img)
## 将图像img从BGR色彩空间转换到灰度空间，转换后的结果保存在img_gray中
image_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
image_gray = img
# 创建小图，2行2列，第二个
plt.subplot(222)
# 颜色参数映射
plt.imshow(img_gray,cmap='gray')
print ("---image gray---")
print (img_gray)

#二值化
rows,cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray[i, j] <= 0.5):
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
img_binary = np.where(img_gray >= 0.5 ,1,0)
print ("---image_binary---")
print(img_gray)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
plt.show()
