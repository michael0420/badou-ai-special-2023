
import cv2

import random

def Gauss_noise(img,percentage):
    w,h = img.shape[:2]
    noise_img = img.copy()       #不带.copy()会对原图像进行修改
    noise_num = int(w*h*percentage)
    for i in range(noise_num):
        X = random.randint(0,w-1)   #random.randint()函数生成的数包含0和w
        Y = random.randint(0,h-1)
        if random.random() <= 0.5:
            noise_img[X,Y] = 0
        else :
            noise_img[X,Y] = 255
    return noise_img


img = cv2.imread(r'F:\badouai\lenna.png')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
noiseimg = Gauss_noise(img_gray,0.2)


cv2.imshow('source',img_gray)
cv2.imshow('spiced_salt',noiseimg)
cv2.waitKey()
cv2.destroyAllWindows()
