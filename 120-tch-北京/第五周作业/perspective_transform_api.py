import cv2
import numpy as np

img = cv2.imread(r'F:\badouai\IMG_4432.png')

input = img.copy()

src = np.float32([[1345, 988], [2740, 1640], [1643, 3261], [121, 1941]])
dst = np.float32([[0, 0], [600, 0], [600, 600], [0, 600]])
print(img.shape)

m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(input, m, (600, 600))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)