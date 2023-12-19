import cv2
import numpy as np


def nearest(img, size):
    height, width, channels = img.shape
    dst_h, dst_w = size
    scale_h = dst_h / height
    scale_w = dst_w / width
    dst_img = np.zeros([dst_h, dst_w, channels], img.dtype)
    for h in range(dst_h):
        for w in range(dst_w):
            # 勿忘取整
            x = int(h / scale_h + 0.5)
            y = int(w / scale_w + 0.5)
            dst_img[h, w] = img[x, y]
    return dst_img


if __name__ == '__main__':
    srcImg = cv2.imread("lenna.png")
    zoom = nearest(srcImg, [800, 800])
    cv2.imshow("lenna", zoom)
    cv2.waitKey()

