import numpy as np
import cv2


def bilinear_interpolation(img,out_dim):
    src_h,src_w,channel = img.shape
    dst_h,dst_w = out_dim[1],out_dim[2]

    if src_h == dst_h and src_w == dst_w:
        return  img.copy()
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)

    scale_x,scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                scx_x0 = int(np.floor(src_x))
                scx_x1 = min(scx_x0+1,src_w -1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h -1 )

                temp0 = (scx_x1  - src_x) * img[src_y0,scx_x0,i] + (src_x - scx_x0) * img[src_y0,scx_x1,i]
                temp1 = (scx_x1 - src_x) * img[src_y1, scx_x0, i] + (src_x - scx_x0) * img[src_y1, scx_x1, i]




    return


if __name__ == '__main__':
    img = cv2.imread("../lenna.png")
    dst  = bilinear_interpolation(img,(700,700))
    cv2.imshow("pc1",dst)
    cv2.waitKey ()