#双线性插值
import cv2
img = cv2.imread('lenna.png')
h, w, channels = img.shape
import numpy as np
H, W = 800, 800
img_bilinear = np.zeros([H, W, channels], dtype=np.uint8)
scale_x, scale_y = float(w) / W, float(h) / H
for c in range(channels):
    for y in range(H):
        for x in range(W):
            src_x = (x + 0.5) * scale_x - 0.5
            src_y = (y + 0.5) * scale_y - 0.5
            # src_x = (x + (W - w) / (2 * W)) * scale_x
            # src_y = (y + (H - h) / (2 * H)) * scale_y
            # src_x = x * scale_x
            # src_y = y * scale_y
            x0 = int(np.floor(src_x))
            y0 = int(np.floor(src_y))
            x1 = min(x0 + 1, h - 1)
            y1 = min(y0 + 1, w - 1)
            R1 = (x1 - src_x) * img[y0, x0, c] + (src_x - x0) * img[y0, x1, c]
            R2 = (x1 - src_x) * img[y1, x0, c] + (src_x - x0) * img[y1, x1, c]
            img_bilinear[y, x, c] = int((y1-src_y)*R1 + (src_y-y0)*R2)

cv2.imshow('bilinear interp', img_bilinear)
cv2.waitKey()
