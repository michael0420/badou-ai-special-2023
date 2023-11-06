import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
# 直方图均衡化
hist_dst = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(),256)
plt.show()

cv2.imshow('dst',np.hstack([gray,dst]))
cv2.waitKey(0)
cv2.destroyAllWindows()