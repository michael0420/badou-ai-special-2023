import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('lenna.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray_img)
plt.figure()
plt.subplot(1, 2, 1)
plt.title('gray_img')
plt.hist(gray_img.ravel(), 256, [0, 256])
plt.subplot(1, 2, 2)
plt.title('dst_img')
plt.hist(dst.ravel(), 256, [0, 256])
plt.show()

cv2.imshow('dst', np.hstack([gray_img, dst]))
cv2.waitKey(0)
