import numpy as np
import cv2

img = cv2.imread('lenna.png')

h = img.shape[0]
w = img.shape[1]
ch = img.shape[2]

dstw = 600
dsth = 600
dst_img = np.zeros((dstw, dsth, 3), dtype=np.uint8)
for i in range(3):
    for dsty in range(h):
        for dstx in range(w):
            srcx = (dstx + 0.5) * (float(w) / dstw) - 0.5
            srcy = (dsty + 0.5) * (float(h) / dsth) - 0.5

            x0 = int(np.floor(srcx))
            x1 = min(x0 + 1, w - 1)
            y0 = int(np.floor(srcy))
            y1 = min(y0 + 1, h - 1)

            f_R1 = (x1 - srcx) * img[y0, x0, i] + (srcx - x0) * img[y0, x1, i]
            f_R2 = (x1 - srcx) * img[y1, x0, i] + (srcx - x0) * img[y1, x1, i]

            dst_img[dsty, dstx, i] = int((y1 - srcy) * f_R1 + (srcy - y0) * f_R2)

cv2.imshow('bilinear interp', dst_img)
cv2.waitKey(0)
