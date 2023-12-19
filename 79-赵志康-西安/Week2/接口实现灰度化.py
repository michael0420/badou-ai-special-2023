from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("123.png")
# plt.imread和cv2.imread区别：
# cv2读取的是bgr图层，转换的话要使用cvtColor(img, cv2.COLOR_BGR2RGB)
# plt读取的直接是图像的数组
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
plt.subplot(221)  # plt.subplot(221)表示将整个图像窗口分为2行2列, 当前位置为1.
plt.imshow(img)  # 将数组的值以图片的形式展示出来
print("---image ori----")
print(img)

# 灰度化
img_gray = rgb2gray(img)
plt.subplot(222)  # plt.subplot(222)表示将整个图像窗口分为2行2列, 当前位置为2.
plt.imshow(img_gray, cmap='gray')  # cmap='gray'显示恢复图像
print("---image gray----")
print(img_gray)

img_binary = np.where(img_gray >= 0.5, 1, 0)
# np.where(condition,x,y) 当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回x，当条件不成立时where返回y;np.where(condition)
# 当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式
print("-----imge_binary------")
print(img_binary)

plt.subplot(223)  # plt.subplot(223)表示将整个图像窗口分为2行2列, 当前位置为3.
plt.imshow(img_binary, cmap='gray')
plt.show()
