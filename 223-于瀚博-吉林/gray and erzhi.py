import cv2
img=cv2.imread('lenna.png')
"img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)"
'''cv2.imshow('gray',img_gray)
cv2.waitKey(0)'''
th1=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,5)
cv2.imshow('th',th1)
cv2.waitKey(0)


