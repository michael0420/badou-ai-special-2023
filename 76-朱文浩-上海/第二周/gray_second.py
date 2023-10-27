import cv2

img = cv2.imread('img.png')
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 二值化 （127，255）
retval,bit_img = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
cv2.imshow('photo1',img)
cv2.imshow('photo2',gray_img)
cv2.imshow('photo3',bit_img)
cv2.waitKey(0)