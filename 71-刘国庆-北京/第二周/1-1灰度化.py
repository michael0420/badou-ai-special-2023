import cv2
import numpy as np

img = cv2.imread("lenna.png")
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3
print("image show gray:\n", img_gray)
cv2.imshow("img_gray", img_gray)
cv2.waitKey(0)
