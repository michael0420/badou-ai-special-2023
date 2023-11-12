import cv2
import numpy as np

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

equ = cv2.equalizeHist(gray)
cv2.imshow('equalized image', equ)
cv2.imshow('origin', img)
cv2.imshow('gamma', gray)
cv2.waitKey(0)
