
import cv2

import random

def Gauss_noise(img,means,sigma,percentage):
    w,h = img.shape[:2]
    noise_img = img.copy()
    noise_num = int(w*h*percentage)
    for i in range(noise_num):
        X = random.randint(0,w-1)   #random.randint()函数生成的数包含0和w
        Y = random.randint(0,h-1)
        noise_img[X,Y] = noise_img[X,Y]+random.gauss(means,sigma)
        if noise_img[X,Y] < 0:             #noise_img[X, Y] = min(max(int(noise_img[X, Y] + noise), 0), 255)
            noise_img[X,Y] = 0
        elif noise_img[X,Y] > 255:
            noise_img[X,Y] = 255
    return noise_img


img = cv2.imread(r'F:\badouai\lenna.png')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
noiseimg = Gauss_noise(img_gray,10,4,0.9)

cv2.imshow('source',img_gray)
cv2.imshow('gauss',noiseimg)
cv2.waitKey()

