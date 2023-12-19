import cv2
import numpy as np


def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape  # 原图的高和宽
    dst_h, dst_w = out_dim[1], out_dim[0]  # 目标图片的高和宽
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)  # 构建一个空的矩阵，空图片
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h  # 计算缩放比例
    # 打印出图像中的所有像素点
    for i in range(3):  # 遍历3个通道
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 找出原图片的坐标x和y
                # 使用几何中心对称
                # 如果直接使用方法，src_x = dst_x * scale_y
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 找原图x和y的两边最近的整数坐标
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # ⭐计算插值（套公式）,img的取值顺序先行后列，先y后x ！！！⭐
                # ⭐图像读取出来的数组形式就是先行后列，所以访问像素的时候只能这样⭐
                # 在x方向上做插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                # 在y方向上做插值
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('bilinear interpolation', dst)
    cv2.waitKey()
