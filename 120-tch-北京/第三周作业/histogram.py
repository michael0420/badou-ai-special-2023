from matplotlib import pyplot as plt
import cv2
import numpy as np

img = cv2.imread(r'F:\badouai\lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = cv2.equalizeHist(gray)

plt.figure()
plt.hist(dst.ravel(),256)    
plt.show()

cv2.imshow('lenna',np.hstack([gray, dst]))
cv2.waitKey()


