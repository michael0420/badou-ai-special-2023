import  cv2
import numpy as np
img=cv2.imread("scenery.png")#读取彩色图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#彩色图转为灰度图
#显示灰度图
cv2.imshow('Image',gray_img)
cv2.waitKey(0)
# print(img)
#二值化
binary_img = np.where(gray_img >= 128, 255, 0)
cv2.imshow('Binary Image',binary_img.astype(np.uint8))
#
cv2.waitKey(0)


