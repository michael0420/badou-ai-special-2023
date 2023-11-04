"""
图像的灰度化和二值化
"""

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from  PIL import Image
import cv2

#实现灰度化
imageSrc = cv2.imread("lenna.png")
height,width = imageSrc.shape[:2] #获取图片的宽高
image_gray = np.zeros([height,width],imageSrc.dtype) #新建一张宽高一致的图片

#遍历原图片的值
for h in range(height):
    for w in range(width):
        srcValue = imageSrc[h,w] #获取原图中的BGR坐标
        image_gray[h,w] = int(srcValue[0]*0.11 + srcValue[1]*0.59 + srcValue[2]*0.3)
print(image_gray)
print("image show gray: %s"%image_gray)
cv2.imshow("image show gray",image_gray)
cv2.waitKey(0)

plt.subplot(221)
# img = plt.imread("lenna.png")
plt.imshow(image_gray,cmap='gray')
plt.show()
print("---image lenna---")
print(img)

#灰度操作
img = plt.imread("lenna.png")
# img_gray = rgb2gray(img)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
plt.show()
print("----image gray----")
print(img_gray)

#二值化
img = plt.imread("lenna.png")
img_binary = rgb2gray(img)
rows, cols = img_binary.shape
for i in range(rows):
    for j in range(cols):
        if (img_binary[i,j] <= 0.5):
            img_binary[i,j] = 0
        else:
            img_binary[i,j] = 1

# img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----image binary")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()