import cv2
import numpy as np
def interpImage(img):
    height, width, channel = img.shape
    emptyImage = np.zeros((800, 800, channel), np.uint8)
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            y = int(j / sw + 0.5)
            emptyImage[i, j] = img[i, j]
    return emptyImage

img = cv2.imread("lenna.png")
zoom = interpImage(img)
print(zoom)
print(zoom.shape)
cv2.imshow("Nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)

