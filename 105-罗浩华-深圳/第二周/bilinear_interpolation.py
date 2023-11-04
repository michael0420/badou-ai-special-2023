import numpy as np
import cv2


def bilinear_interpolation(img, input):
    src_height, src_width, channel = img.shape
    print('src_height:', src_height, 'src_width:', src_width, 'channel:', channel)
    new_height, new_width = input[1],input[0]
    print('new_height:', new_height, 'new_width:', new_width)
    if src_height == new_height and src_width == new_width:
        return img.copy()

    new_img = np.zeros((new_height, new_width, 3), dtype=img.dtype)
    s_height, s_width = float (src_height) / new_height, float (src_width) / new_width

    for i in range(channel):
        for cur_height in range(new_height):
            for cur_width in range(new_width):
                src_x = (cur_width + 0.5) * s_width - 0.5
                src_y = (cur_height + 0.5) * s_height - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_width - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_height - 1)

                point0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                point1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                new_img[cur_height, cur_width, i] = int((src_y1 - src_y) * point0 + (src_y - src_y0) * point1)
    return new_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (900, 900))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()
