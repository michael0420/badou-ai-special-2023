import cv2
img = cv2.imread("photo.jpg")
img_ = img.copy()

cv2.imshow("photo", img)
cv2.waitKey()