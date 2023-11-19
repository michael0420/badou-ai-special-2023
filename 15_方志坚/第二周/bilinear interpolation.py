#!/usr/bin/python
# -*- coding: utf-8 -*-
 
import numpy as np
import cv2
 
'''
python implementation of bilinear interpolation
''' 
def bilinear_interpolation(img,out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
        # Checks if the input image dimensions are already equal to the desired output dimensions.
        # If they are, the function returns a copy of the input image
        # as there is no need to perform interpolation.
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    # this ratio note can be found from the ppt 04.

    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
 
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                # this equation is fixed and proved, no need to prove it again. also in the ppt 04 file.
                src_x = (dst_x + 0.5) * scale_x-0.5
                src_y = (dst_y + 0.5) * scale_y-0.5
 
                # find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x)) # floor function to reduce the number from many to 0 after decimal
                src_x1 = min(src_x0 + 1 ,src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                # src_x0 and src_y0 are the floor values of the source coordinates src_x and src_y. They represent the
                # top-left pixel in the source image that surrounds the desired location in the destination image.

                # src_x1 and src_y1 are calculated by taking src_x0 + 1 and src_y0 + 1, respectively. These represent
                # the bottom-right pixel in the source image that surrounds the desired location in the destination image.
                #
                # min(src_x0 + 1, src_w - 1) and min(src_y0 + 1, src_h - 1) ensure that the indices do not go beyond the
                # boundaries of the source image. If the calculated values exceed the maximum indices
                # (src_w - 1 for the width and src_h - 1 for the height), they are capped to the maximum allowable values.

 
                # calculate the interpolation
                # this is the formula from ppt 04 and it is fixed.
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
 
    return dst_img
 
# this bilinear_interpolation method will make the image looks more smoothing if you enlarge the image.
# but it includes more computation.

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img,(700,700))
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey(0)
