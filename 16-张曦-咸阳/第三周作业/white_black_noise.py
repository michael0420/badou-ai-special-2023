import random

import cv2

def white_black_noise(img, percent):
    noise_img = img
    h, w = noise_img.shape
    randomNum = int(h * w * percent)
    for i in range(0, randomNum):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        if random.randint(1,2) == 1:
            noise_img[x, y] = 0
        else :
            noise_img[x, y] = 255
    return noise_img

image = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAY", img_gray)
noise_image = white_black_noise(img_gray, 0.1)
cv2.imshow("NOISE", noise_image)

cv2.waitKey(0)