import cv2
import numpy as np

gray = cv2.imread("lenna.png", 0)
cv2.imshow("canny", cv2.Canny(gray, 200, 300))

cv2.waitKey()
cv2.destroyAllWindows()