import numpy as np
import cv2


def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    # 分别求出缩放比
    m_w = src_w / dst_w
    m_h = src_h / dst_h
    # 循环每个通道的每个像素点坐标
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 中心对齐情况下dst_img在img坐标系下的坐标
                src_x = (dst_x + 0.5) * m_w - 0.5
                src_y = (dst_y + 0.5) * m_h - 0.5
                # 找到映射到原图的虚拟点四周真实的像素点坐标
                # 以下是四个点坐标（src_x0，src_y0），（src_x1，src_y0），（src_x0，src_y1），（src_x1，src_y1）
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)  # 防止越界
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)  # 防止越界
                # 算出该点像素值，代入公式
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img


if __name__ == '__main__':
    img = cv2.imread('D:/cv_project/demo_1/lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('bilibili', dst)
    cv2.waitKey()
