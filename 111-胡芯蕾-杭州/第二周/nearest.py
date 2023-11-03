import cv2
import numpy as np
import matplotlib.pyplot as plt


# 最邻近插值算法
def nearest(img, dst_h, dst_w):
    img_h, img_w, channels = img.shape
    dst = np.zeros((dst_h, dst_w, channels), np.uint8)
    sh = dst_h / img_h
    sw = dst_w / img_w
    for i in range(dst_h):
        for j in range(dst_w):
            x = int(i / sh + 0.5)  # int(),转为整型，使用向下取整，+0.5 转为四舍五入。
            y = int(j / sw + 0.5)
            dst[i, j] = img[x, y]
    return dst


if __name__ == '__main__':
    image = cv2.imread("lenna.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dst_image = nearest(image, 800, 800)
    print(dst_image)
    print(dst_image.shape)

    dst_image1 = cv2.resize(image, (800, 800), interpolation=cv2.INTER_NEAREST)

    plt.subplot(131), plt.axis('off'), plt.title("image")
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.title("nearest"), plt.axis('off')
    plt.imshow(dst_image, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.title("cv-nearest"), plt.axis('off')
    plt.imshow(dst_image1, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
