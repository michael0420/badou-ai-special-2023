import cv2
import numpy as np


img = cv2.imread(r"F:\badouai\lenna.png")
h,w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)   #可以将数值除以255，将RGB值转化为0-1的浮点数

rows, cols = img_gray.shape
img_binary = np.zeros([rows,cols],img.dtype)
for i in range(rows):
    for j in range(cols):
        if (img_gray[i, j] <= 127):
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 255

print (img_gray)
print("image show gray: %s"%img_gray)
print("image show binary:%s"%img_binary)
cv2.imshow("image show gray",img_gray)
cv2.imshow("image show binary",img_binary)
cv2.waitKey()
#cv2.destroyAllWindows()