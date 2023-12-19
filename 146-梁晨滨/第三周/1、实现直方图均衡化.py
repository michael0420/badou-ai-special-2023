import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = [u'SimHei']

def Histogram_Equalization(img):
    H, W, C = img.shape
    total_pixel = H * W
    # grey_probability放置灰度级,0-255共256个灰度级
    grey_probability = [0] * 256
    img_grey = np.zeros([H, W], img.dtype)
    img_output = np.zeros([H, W], img.dtype)

    # 统计所有像素点灰度级
    for i in range(H):
        for j in range(W):
            # 三通道融合有小数，所以加0.5四舍五入
            img_grey[i, j] = int(img[i, j, 0] * 0.11 + img[i, j, 1] * 0.59 + img[i, j, 2] * 0.3 + 0.5)
            # 灰度级需要整数
            grey = round(img_grey[i, j])
            grey_probability[grey] += 1

    # min_grey, max_grey = min(grey_probability), max(grey_probability)
    # 计算每个灰度级的概率,并进行累加概率计算
    for i in range(256):
        grey_probability[i] /= total_pixel
        grey_probability[i] += grey_probability[i - 1]

    # 还原到0-255的灰度上
    for i in range(256):
        grey_probability[i] = grey_probability[i] * 256 - 1

    # 对灰度图进行重新赋值
    for i in range(H):
        for j in range(W):
            img_output[i, j] = grey_probability[img_grey[i, j]]


    return img_grey, img_output


if __name__ == '__main__':
    img_input = plt.imread("lenna.png")
    img_input = (img_input * 255).astype(np.uint8)
    img_grey, img_output = Histogram_Equalization(img_input)

    plt.subplot(131), plt.title("原图")
    plt.imshow(img_input)

    plt.subplot(132), plt.title("原图灰度图")
    plt.imshow(img_grey, cmap='gray')

    plt.subplot(133), plt.title("均衡化后的图")
    plt.imshow(img_output, cmap='gray')
    plt.show()

