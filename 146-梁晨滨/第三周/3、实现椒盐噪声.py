import random
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = [u'SimHei']


def GaussianNoise(img, percetage):
    H, W, C = img.shape
    noise_pixel = int(H * W * percetage)
    # img_noise彩色图加噪，img_grey原始图片的灰度图，img_grey_noise灰度图像加噪
    img_noise = np.array(list(img))
    img_grey = np.zeros([H, W], img.dtype)

    for i in range(H):
        for j in range(W):
            # 三通道融合变灰度
            img_grey[i, j] = img[i, j, 0] * 0.11 + img[i, j, 1] * 0.59 + img[i, j, 2] * 0.3 + 0.5

    img_grey_noise = np.array(list(img_grey))

    # 彩色图加噪
    for c in range(C):
        for i in range(noise_pixel):
            # 计算出加噪声的坐标(x_noise, y_noise)
            x_noise = random.randint(0, W - 1)
            y_noise = random.randint(0, H - 1)

            # 随机椒盐是0还是255
            if random.random() < 0.5:
                noise = 0
            else:
                noise = 255

            # 原图像随机点替换为噪声
            img_noise[y_noise, x_noise, c] = noise

    # 灰度图加噪
    for i in range(noise_pixel):
        # 计算出加噪声的坐标(x_noise, y_noise)
        x_noise = random.randint(0, W - 1)
        y_noise = random.randint(0, H - 1)

        # 随机椒盐是0还是255
        if random.random() < 0.5:
            noise = 0
        else:
            noise = 255

        # 原图像随机点替换为噪声
        img_grey_noise[y_noise, x_noise] = noise

    return img_noise, img_grey, img_grey_noise


if __name__ == '__main__':
    img_input = plt.imread("lenna.png")
    img_input = (img_input * 255).astype(np.uint8)
    img_noise, img_grey, img_grey_noise = GaussianNoise(img_input, 0.3)

    plt.subplots_adjust(hspace=0.5)
    plt.subplot(221), plt.title("原图")
    plt.imshow(img_input)

    plt.subplot(222), plt.title("原图+椒盐")
    plt.imshow(img_noise, cmap='gray')

    plt.subplot(223), plt.title("原图灰度")
    plt.imshow(img_grey, cmap='gray')

    plt.subplot(224), plt.title("原图灰度+椒盐")
    plt.imshow(img_grey_noise, cmap='gray')
    plt.show()

