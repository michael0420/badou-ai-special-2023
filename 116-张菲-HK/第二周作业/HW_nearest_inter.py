import cv2
import numpy as np
import matplotlib.pyplot as plt


def nearest_inter(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage


img = cv2.imread("lenna.png")
# img = plt.imread("C:/Users/User/Desktop/Ai/八斗2023AI精品班/【2】数学&数字图像/代码/lenna.png")
# img = img[:, :, :: -1]
new_img = nearest_inter(img)
print(new_img)
print(new_img.shape)
cv2.imshow("nearest interp", new_img)
cv2.imshow("image", img)
cv2.waitKey(0)

