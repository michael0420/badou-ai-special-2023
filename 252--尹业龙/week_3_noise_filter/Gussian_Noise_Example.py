"""
高斯噪声算法测试用例
Gussian_Noise_Example
data:2023.9.13
"""
import cv2
from Gussian_Noise import gussian_noise

sigma = 2
mean = 4
percentage = 0.8
img = cv2.imread("D:\\subject_learning\\cv_learn\\project\\lenna.jpg", 0)
cv2.imshow('img', img)
noise_img = gussian_noise(img, sigma, mean, percentage)
cv2.imshow('noise_img', noise_img)
cv2.waitKey(0)


