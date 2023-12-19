import numpy as np
import cv2

if __name__ == "__main__":
    # 读取原图，赋予变量img
    img = cv2.imread("C:/Users/15082/Desktop/lenna.png")
    cv2.imshow("lenna", img)

    # 将原图处理为灰度图，赋予变量grey_img
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grey", grey_img)

    # 读取灰度图的高和宽，赋予变量height和width
    height, width = img.shape[:2]

    # 创建高斯噪声，赋予变量gaussian_noise
    means = 0
    std = 25
    gussian_noise = np.random.normal(means, std, (height, width))

    noise_img = np.clip(grey_img + gussian_noise, 0, 255).astype(np.uint8)
    cv2.imshow("noise", noise_img)
    cv2.waitKey(0)
