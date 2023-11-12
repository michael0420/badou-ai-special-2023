from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
img = cv2.imread("lenna.png")
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)

for i in range(h):
    for j in range(w):
        m=img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)
cv2.waitKey()

#二值化
for i in range(h):
    for j in range(w):
        if img_gray[i,j]/255 <= 0.5:
            img_gray[i,j] = 0
        else:
            img_gray[i,j] = 1
print("image show gray: %s"%img_gray)
plt.subplot(111)
plt.imshow(img_gray, cmap='gray')
plt.show()








