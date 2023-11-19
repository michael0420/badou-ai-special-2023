import cv2
import numpy as np
img = cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img,30,100)
cv2.imshow('img', img)
cv2.imshow('Canny', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()