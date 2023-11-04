import cv2
img = cv2.imread('lenna.png')
h, w = img.shape[0:2]
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_SP = img_gray.copy()
percentage = 0.2
import random
for i in range(int(percentage*h*w)):
    randx = random.randint(1, h-1)
    randy = random.randint(1, w-1)
    if random.random() <= 0.5:
        img_SP[randx][randy] = 0
    else:
        img_SP[randx][randy] = 255
import numpy as np
cv2.imshow('Salt-and-pepper noise', np.hstack([img_gray, img_SP]))
cv2.waitKey()
