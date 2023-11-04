import numpy as np
import cv2
import matplotlib.pyplot as plt


# 双线性插值算法
def bilinear(img, out_dim):
    img_h, img_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("img_h, img_w = ", img_h, img_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if img_h == dst_h and img_w == dst_w:
        return img.copy()
    dst_image = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(img_w) / dst_w, float(img_h) / dst_h
    for i in range(channel):
        for dstY in range(dst_h):
            for dstX in range(dst_w):
                # 中心几何对称
                # 直接使用时 src_x = dst_x * scale_x
                dstX_srcX = (dstX + 0.5) * scale_x - 0.5
                dstY_srcY = (dstY + 0.5) * scale_y - 0.5

                # 找出将用于计算插值的点的坐标
                src_x0 = int(np.floor(dstX_srcX))
                src_x1 = min(src_x0 + 1, img_w - 1)
                src_y0 = int(np.floor(dstY_srcY))
                src_y1 = min(src_y0 + 1, img_h - 1)

                # calculate the interpolation
                temp0 = (src_x1 - dstX_srcX) * img[src_y0, src_x0, i] + (dstX_srcX - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - dstX_srcX) * img[src_y1, src_x0, i] + (dstX_srcX - src_x0) * img[src_y1, src_x1, i]
                dst_image[dstY, dstX, i] = int((src_y1 - dstY_srcY) * temp0 + (dstY_srcY - src_y0) * temp1)

    return dst_image


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = bilinear(img, (700, 700))

    dst_1 = cv2.resize(img, (700, 700))

    plt.subplot(131), plt.axis('off'), plt.title("image")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.title("bilinear"), plt.axis('off')
    plt.imshow(dst, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.title("cv-bilinear"), plt.axis('off')
    plt.imshow(dst_1, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
