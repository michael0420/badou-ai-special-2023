import cv2
import numpy as np

img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("canny", cv2.Canny(gray, 50, 60))
cv2.waitKey()
cv2.destroyAllWindows()
