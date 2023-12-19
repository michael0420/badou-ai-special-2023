import cv2
import numpy as np

img = cv2.imread('test_images.jpeg')

new_img = img.copy()

src = np.float32([[87, 44], [175, 78], [161, 136], [64, 103]])
dst = np.float32([[0, 0], [300, 0], [300, 165], [0, 165]])

# wrapmatrix
mat = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(mat)


res_plot = cv2.warpPerspective(new_img, mat, (300, 165))
cv2.imshow("src", img)
cv2.imshow("result", res_plot)
cv2.waitKey(0)