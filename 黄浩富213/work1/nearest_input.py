import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_nearest_inter(imge):
    img_source = cv2.imread(imge)
    h = img_source.shape[0]
    w = img_source.shape[1]
    ch = img_source.shape[2]
    print(h, w, ch)
    img_new = np.zeros((640, 800, ch), dtype=np.uint8)
    for i in range(640):
        for j in range(800):
            srcx = int((i * (h/ 640)) + 0.5)
            srcy = int((j * (w / 800)) + 0.5)
            img_new[i, j] = img_source[srcx, srcy]
    cv2.imshow('lenna_new', img_new)
    cv2.imshow('lenna', img_source)
    cv2.waitKey(0)

if __name__ == '__main__':
    get_nearest_inter('lenna.png')