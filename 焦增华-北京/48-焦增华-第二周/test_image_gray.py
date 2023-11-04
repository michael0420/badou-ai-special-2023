import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import  rgb2gray

# 灰度化
img = cv2.imread("../lenna.png")
h,w = img.shape[:2]   #获取图片的high 和weight
img_gray = np.zeros([h,w],img.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for k in range(w):
        m = img[i,k]  #取出当前h  w 的BGR坐标
        img_gray[i,k] = int(m[0]*0.11 + m[1]*0.59 + m[2]* 0.3)
print(img_gray)
plt.subplot(224)
plt.imshow(img_gray,cmap='gray')
 #设置布局  分割为二行二列  图像位于第一个


plt.subplot(221)  #设置布局  分割为二行二列  图像位于第一个
img = plt.imread("../lenna.png")
# img = cv2.imread("lenna.png", False)
plt.imshow(img)
print("---image lenna----")
print(img)

# 灰度化
# img_gray_01 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img_gray_01 = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray_01, cmap='gray')

# rows,cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if(img_gray[i,j] <= 4):
#             img[i,j] = 20
#         else:
#             img[i, j] = 10


# 二值化
img_binary = np.where(img_gray_01 >= 0.5,1,0)
print(img_binary)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()









