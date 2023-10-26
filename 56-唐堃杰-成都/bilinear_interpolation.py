import cv2
import numpy
import numpy as np


def get_bilinear(cur_img, dir):
    src_h, src_w, channels = cur_img.shape
    dst_h, dst_w = dir[0],dir[1]
    new_img = numpy.zeros([dst_h, dst_w, 3], cur_img.dtype)
    x_radio, y_radio = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for j in range(dst_h):
            for k in range(dst_w):
                # 中心重合
                src_x = (k + 0.5) * x_radio - 0.5
                src_y = (j + 0.5) * y_radio - 0.5

                # 找到对应插值点的坐标
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h -1)

                # 套用公式进行双线插值
                tmp =  (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                tmp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                new_img[j, k, i] = int((src_y1 - src_y) * tmp + (src_y - src_y0) * tmp1)
    return new_img


if __name__ == "__main__":
    img = cv2.imread("lenna.png")
    data = get_bilinear(img, [800, 800])
    print(data)
    cv2.waitKey()