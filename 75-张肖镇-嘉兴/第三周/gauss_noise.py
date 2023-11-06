import cv2 as cv
import numpy as np


def gasuss_noise(image, mu=0.0, sigma=0.1):
    """
     添加高斯噪声
    :param image: 输入的图像
    :param mu: 均值
    :param sigma: 标准差
    :return: 含有高斯噪声的图像
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mu, sigma, image.shape)
    gauss_noise = image + noise
    if gauss_noise.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    gauss_noise = np.clip(gauss_noise, low_clip, 1)
    gauss_noise = gauss_noise * 255
    gauss_noise = np.uint8(gauss_noise)
    return gauss_noise


if __name__ == '__main__':
    # ----------------------读取图片-----------------------------
    img = cv.imread("test.jpg")
    # --------------------添加高斯噪声---------------------------
    out2 = gasuss_noise(img, mu=0.0, sigma=0.1)
    # ----------------------显示结果-----------------------------
    cv.imshow('origion_pic', img)
    cv.imshow('gauss_noise', out2)
    cv.waitKey(0)