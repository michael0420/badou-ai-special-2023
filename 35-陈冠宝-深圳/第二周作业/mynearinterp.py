import cv2
import numpy as np
def function(img, dsth, dstw):
    height, width, channels = img.shape
    emptyImage = np.zeros((dsth, dstw, channels), np.uint8)
    sh = dsth / height
    sw = dstw / width
    for i in range(dsth):
        for j in range(dstw):
            x = int(i / sh + 0.5)  # int(),转为整型，使用向下取整。
            y = int(j / sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage

img = cv2.imread("lenna.png")
zoom = function(img, 800, 500)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)


