import random
import cv2
img = cv2.imread('lenna.png')
h, w = img.shape[0:2]
img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_Gaussian = img_Gray.copy()
means = 2
sigma = 4
#给每个像素点加高斯噪声
# for i in range(h):
#     for j in range(w):
#         gauss = random.gauss(means, sigma)
#         if img_Gaussian[i][j] + gauss > 255:
#             img_Gaussian[i][j] = 255
#         elif img_Gaussian[i][j] + gauss < 0:
#             img_Gaussian[i][j] = 0
#         else:
#             img_Gaussian[i][j] += gauss
#给部分像素点加高斯噪声
percentage = 0.8
import numpy as np
img = cv2.imread('lenna.png')
img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_Gaussian = img_Gray.copy()
for i in range(int(percentage*h*w)):
    randx = random.randint(1, h-1)
    randy = random.randint(1, w-1)
    gauss = random.gauss(means, sigma)
    if img_Gaussian[randx][randy] + gauss > 255:
        img_Gaussian[randx][randy] = 255
    elif img_Gaussian[randx][randy] + gauss < 0:
        img_Gaussian[randx][randy] = 0
    else:
        img_Gaussian[randx][randy] += gauss
import numpy as np
cv2.imshow('Gaussian noise', np.hstack([img_Gray, img_Gaussian]))
cv2.waitKey()
