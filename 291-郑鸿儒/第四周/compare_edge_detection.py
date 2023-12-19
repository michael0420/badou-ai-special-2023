#!/usr/bin/env python
# encoding=utf-8
import cv2
import matplotlib.pyplot as plt

gray = cv2.imread("img/lenna.png", 0)

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

laplace = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
canny = cv2.Canny(gray, 100, 200)
plt.figure()
plt.subplot(231), plt.imshow(gray, "gray"), plt.title('origin')
plt.subplot(232), plt.imshow(sobel_x, "gray"), plt.title('sobel_x')
plt.subplot(233), plt.imshow(sobel_y, 'gray'), plt.title('sobel_y')
plt.subplot(234), plt.imshow(laplace, 'gray'), plt.title('laplace')
plt.subplot(235), plt.imshow(canny, 'gray'), plt.title('canny')
plt.axis('off')
plt.show()
