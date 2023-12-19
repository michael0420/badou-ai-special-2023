import cv2
import numpy as np


def binary(img, size):
    dst_h, dst_w = size
    height, width, channels = img.shape
    dst_img = np.zeros([dst_h, dst_w, channels], img.dtype)
    scale_x = dst_w / width
    scale_y = dst_h / height
    print(scale_y)
    for c in range(channels):
        for h in range(dst_h):
            for w in range(dst_w):
                src_x = (w + 0.5) / scale_x - 0.5
                src_y = (h + 0.5) / scale_y - 0.5
                src_x0 = int(src_x)
                src_y0 = int(src_y)
                src_x1 = min(src_x0 + 1, width - 1)
                src_y1 = min(src_y0 + 1, height - 1)
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, c] + (src_x - src_x0) * img[src_y0, src_x1, c]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, c] + (src_x - src_x0) * img[src_y1, src_x1, c]
                dst_img[h, w, c] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img


if "__main__" == __name__:
    src_img = cv2.imread("lenna.png")
    zoom = binary(src_img, [800, 800])
    cv2.imshow("lenna", zoom)
    cv2.waitKey()
