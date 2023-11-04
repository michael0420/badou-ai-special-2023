import cv2
import numpy as np

def bilinear_interpolation(img, out_dim):
    src_h, src_w, chanels= img.shape
    dst_h, dst_w = out_dim[0], out_dim[1]
    if src_h == dst_h and src_w == dst_w:
        print('图片尺寸相同')
        return img.copy()

    dst_img = np.zeros([dst_h, dst_w, 3], dtype=np.uint8)
    scale_x = float(src_w) / dst_w
    scale_y = float(src_h) / dst_h

    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                x = (dst_x + 0.5) * scale_x - 0.5
                y = (dst_y + 0.5) * scale_y - 0.5
                #x = dst_x * scale_x
                #y = dst_y * scale_y

                x0 = int(np.floor(x))
                y0 = int(np.floor(y))

                x1 = min(x0 + 1, src_w - 1)
                y1 = min(y0 + 1, src_h - 1)

                R1 = (x1 - x) * img[y0, x0, i] + (x - x0) * img[y0, x1, i]
                R2 = (x1 - x) * img[y1, x0, i] + (x - x0) * img[y1, x1, i]

                dst_img[dst_y, dst_x, i] = int( (y1 - y ) * R1 + (y - y0) * R2 )

    return  dst_img

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (600,600))
    cv2.imshow('shuo-learning', dst)
    cv2.waitKey(0)