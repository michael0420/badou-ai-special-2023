from matplotlib import pyplot as plt
import numpy as np
import cv2

# 双线性插值算法
def bilinear_interpolation(src_image_vector, dst_height_times, dst_width_times):
    srcHeight, srcWidth, channels = src_image_vector.shape
    dstHeight = dst_height_times
    dstWidth = dst_width_times
    if srcHeight == dstHeight and srcWidth == dstWidth:
        return src_image_vector.copy()
    dstVector = np.zeros((dstHeight, dstWidth, channels), dtype=np.uint8)
    scale_x, scale_y = float(srcWidth) / dstWidth, float(srcHeight) / dstHeight
    for i in range(channels):
        for dstY in range(dstHeight):
            for dstX in range(dstWidth):
                # dstX_srcX = dstX * (srcWidth / dstWidth)
                # dstY_srcY = dstY * (srcHeight / dstHeight)
                dstX_srcX = (dstX+0.5)*scale_x-0.5
                dstY_srcY = (dstY+0.5)*scale_y-0.5

                src_x0 = int(np.floor(dstX_srcX))
                src_x1 = min(src_x0 + 1, srcWidth - 1)
                src_y0 = int(np.floor(dstY_srcY))
                src_y1 = min(src_y0 + 1, srcHeight - 1)
                temp0 = (src_x1 - dstX_srcX) * src_image_vector[src_y0, src_x0, i] + (dstX_srcX - src_x0) * src_image_vector[src_y0, src_x1, i]
                temp1 = (src_x1 - dstX_srcX) * src_image_vector[src_y1, src_x0, i] + (dstX_srcX - src_x0) * src_image_vector[src_y1, src_x1, i]
                dstVector[dstY, dstX, i] = int((src_y1 - dstY_srcY) * temp0 + (dstY_srcY - src_y0) * temp1)
    return dstVector


if __name__ == "__main__":
    img = cv2.imread('lenna.png')

    result_image = bilinear_interpolation(img, 700, 700)
    cv2.imshow('img', img)
    cv2.imshow('resimg', result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
