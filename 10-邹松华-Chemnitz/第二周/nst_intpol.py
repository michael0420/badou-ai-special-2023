# Homework Assignment / Hausaufgabe vom Songhua Zou

import numpy as np
import cv2

def nst_intpol(img, dst_dim):

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

    # So here's the difference between bilin. intpol. and nst. intpol..
    # With nst. intpol. approach, we assign the value of the nearest pixel to the destination pixel,
    # or it's more like we just copy all the informations (e.g. coordinates, channels, etc.) to the new pixel.
    # We don't need to read the channel or other infos of the source pixel.
    # Just need to assign (copy) the value to the new pixel.

    for dst_y in range(dst_h):
        for dst_x in range(dst_w):
            intpol_y = int(dst_y * scale_h + 0.5)
            intpol_x = int(dst_x * scale_w + 0.5)
            # Note: The addition of +0.5 ensures a better rounding approximation when using int,
            # different from the image center alignment in bilinear interpolation.
            # Also, the calculation for the scaling factor is scale = src/dst, hence src = dst*scale.
            # Don't get it mixed up.

            dst_img[dst_y, dst_x] = img[intpol_y, intpol_x]

    return dst_img

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = nst_intpol(img, (800, 800))
    print(dst)
    print(dst.shape)

    cv2.imshow("ori img", img)
    cv2.imshow("nst intpol", dst)
    cv2.waitKey()
