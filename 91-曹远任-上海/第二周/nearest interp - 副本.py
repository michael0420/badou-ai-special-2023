import cv2
import numpy as np


def function(height, width, img):
    yheight, ywidth, channel = img.shape
    zoomimg = np.zeros((height, width, channel), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            srcx = round((j+0.5) * ywidth / width-0.5)
            srcy = round((i+0.5) * yheight / height-0.5)
            zoomimg[i][j] = img[srcy][srcx]
    return zoomimg


img = cv2.imread("lenna.png")
zoom = function(3000, 3000, img)
print(img.shape, zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)
