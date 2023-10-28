#双线性插值

import  numpy as np
import cv2

def bilinear_interp(img,a):
    src_h,src_w,channel=img.shape
    dst_h,dst_w=a[0],a[1]
    # print("src_h,src)w=",src_h,src_w)
    # print('dst_h,dst_w',dst_h,dst_w)
    if src_h==dst_h and src_w==dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x=(dst_x+0.5)*float(src_w)/dst_w-0.5
                src_y=(dst_y+0.5)*float(src_h)/dst_h-0.5

                src_x0=int(np.floor(src_x))
                src_x1=min(src_x0+1,src_w-1)
                src_y0=int(np.floor(src_y))
                src_y1=min(src_y0+1,src_h-1)

                tempx0=(src_x1-src_x)*img[src_y0,src_x0,i]+(src_x-src_x0)*img[src_y0,src_x1,i]#注意四个点的书写规律
                tempx1=(src_x1-src_x)*img[src_y1,src_x0,i]+(src_x-src_x0)*img[src_y1,src_x1,i]

                dst_img[dst_y,dst_x,i]=int((src_y1-src_y)*tempx0+(src_y-src_y0)*tempx1)

    return dst_img

if __name__=='__main__':
    img=cv2.imread('lenna.png')
    dst=bilinear_interp(img,(800,800))
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey()#运行时间长
