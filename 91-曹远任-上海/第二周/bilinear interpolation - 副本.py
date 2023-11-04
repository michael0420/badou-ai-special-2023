import numpy as np
import cv2


def bilinear_interpolation(height, width, img):
    yheight, ywidth, channel = img.shape
    zoomimg = np.zeros((height, width, channel), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            x = (j + 0.5) * ywidth / width - 0.5
            y = (i + 0.5) * yheight / height - 0.50
            x1 = int(x)
            y1 = int(y)
            x2 = min(x1 + 1, ywidth-1)
            y2 = min(y1 + 1, yheight-1)
            if x1 != x2 and y1 != y2:
                temp1 = (x2 - x) / (x2 - x1) * img[y1][x1] + (x - x1) / (x2 - x1) * img[y1][x2]
                temp2 = (x2 - x) / (x2 - x1) * img[y2][x1] + (x - x1) / (x2 - x1) * img[y2][x2]
                zoomimg[i][j] = (y2 - y) / (y2 - y1) * temp1 + (y - y1) / (y2 - y1) * temp2
    return zoomimg


img = cv2.imread('lenna.png')
dst = bilinear_interpolation(3000, 3000, img)
cv2.imshow('bilinear interp', dst)
cv2.waitKey()
