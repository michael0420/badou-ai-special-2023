# Homework Assignment / Hausaufgabe vom Songhua Zou

import numpy as np
import cv2

def bilin_intpol(img, dst_dim):

    src_h, src_w, channel = img.shape
    # Be careful, func. img.shape returns the height, width and num. of channel
    # Don't mix up the order.

    dst_h, dst_w = dst_dim[1], dst_dim[0]
    # Here's something interesting. With the order of img.shape, we get the height and width.
    # But normally when we describe a resolution of an image, we'll say width * height.
    # So here's the reason, why we use the value of dim[0] for the width,
    # and we use the value of dim[1] for the height.

    # print out all the specs of the source image and destination image.
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)

    # If the specs of source image and destination image are same, then we can direct output the image.
    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    # use zero fill the blank image
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)

    # calculate the scale here
    scale_h, scale_w = float(src_h) / dst_h, float(src_w) / dst_w

    # Loop through all 3 channels and traverse every row and column.
    for num_channel in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                # find out the relationship of geo center
                # between source image and destination image.
                src_y = (dst_y + 0.5) * scale_h - 0.5
                src_x = (dst_x + 0.5) * scale_w - 0.5

                # Find out the relative boundary of the destination image,
                # and prepare for the interpolation.
                # l: left, r: right, t: top, b: bottom
                src_xl = int(np.floor(src_x))
                src_xr = min(src_xl + 1, src_w - 1)
                src_yt = int(np.floor(src_y))
                src_yb = min(src_yt + 1, src_h - 1)

                # compute the interpolation
                intpol_xt = (src_x - src_xl) * img[src_yt, src_xr, num_channel] + (src_xr - src_x) * img[src_yt, src_xl, num_channel]
                intpol_xb = (src_x - src_xl) * img[src_yb, src_xr, num_channel] + (src_xr - src_x) * img[src_yb, src_xl, num_channel]
                dst_img[dst_y, dst_x, num_channel] = int((src_y - src_yt) * intpol_xb + (src_yb - src_y) * intpol_xt)

                # For the compute part, I'd like to say.
                # This approach differs from the standard bilinear interpolation algorithm.
                # It is based on the "inverse proportion of distance" to weight the values of points l and r.
                # In certain contexts, this method is more convenient and efficient.
                # However, this means the interpolation result will slightly differ from the standard bilinear interpolation.

    return dst_img

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    # img = cv2.imread('../代码/lenna.png')
    dst = bilin_intpol(img, (800, 800))
    cv2.imshow('bilinear intpol', dst)
    cv2.waitKey()


                


