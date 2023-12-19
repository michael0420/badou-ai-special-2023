import numpy as np
import cv2
# 灰度化
img = cv2.imread("lenna.png")
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int((m[0]*11 + m[1]*59 + m[2]*30)/100)

cv2.imshow("image show gray", img_gray)
img_graybycv2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #opencv读取BGR图像
#img_graybycv2=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  #opencv读取BGR图像
cv2.imshow("image show graybycv2", img_graybycv2)
cv2.waitKey()