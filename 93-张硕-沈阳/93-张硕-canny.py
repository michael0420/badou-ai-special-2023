import cv2
import numpy as np

img = cv2.imread('lenna.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgcanny = cv2.Canny(gray, 200, 300)
cv2.imshow('canny', imgcanny)
cv2.waitKey(0)
