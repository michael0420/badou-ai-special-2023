import cv2
import numpy as np

img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def CannyThreshold(lowThreshold, ratio = 3, kernel_size = 3):
    img_Gaussian = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_Canny = cv2.Canny(img_Gaussian,
                          lowThreshold,
                          lowThreshold * ratio,
                          apertureSize=kernel_size)
    img_color_canny = cv2.bitwise_and(img, img, mask=img_Canny)
    cv2.imshow("canny demo", img_color_canny)

lowThreshold = 0
max_lowThreshold = 100
cv2.namedWindow("canny demo")
cv2.createTrackbar("Min Threshold", "canny demo", lowThreshold, max_lowThreshold, CannyThreshold)

if cv2.waitKey(0) == 27:  # wait for ESC key to exit cv2
    cv2.destroyAllWindows()
