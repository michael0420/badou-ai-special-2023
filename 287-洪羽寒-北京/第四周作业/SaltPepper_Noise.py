'''
椒盐噪声需求流程：
1.指定信噪比 SNR（信号和噪声所占比例） ，其取值范围在[0, 1]之间
2.计算总像素数目 SP， 得到要加噪的像素数目 NP = SP * SNR
3.随机获取要加噪的每个像素位置P（i, j）
4.指定像素值为255或者0。
5.重复3, 4两个步骤完成所有NP个像素的加噪
'''

import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

'''
1.指定信噪比 SNR（信号和噪声所占比例） ，其取值范围在[0, 1]之间
2.计算总像素数目 SP， 得到要加噪的像素数目 NP = SP * SNR
3.随机获取要加噪的每个像素位置P（i, j）
4.指定像素值为255或者0。
5.重复3, 4两个步骤完成所有NP个像素的加噪
'''

def saltPapper_noise(src_img,percent):                              #src_img输入原图；percent百分比
    a,b,c = src_img.shape
    result_img = np.copy(src_img)
    Z = int(a*b*percent)
    for i in range(Z):
        for j in range(c):
            x = random.randint(0,a-1)                               #首次随机随机生成的行X
            y = random.randint(0,b-1)                               #随机生成的列Y

            if random.random() < 0.5:                               #random.random生成(二次)随机浮点数
                result_img[x,y,j] = 0                               #一个像素点有一半几率是0
            else:
                result_img[x, y, j] = 255                           #另一半几率是255

    return result_img

def salt_noise(src_img,percent):
    a,b,c = src_img.shape
    result_img = np.copy(src_img)
    Z = int(a*b*percent)
    for i in range(Z):
        for j in range(c):
            x = random.randint(0,a-1)
            y = random.randint(0,b-1)
            if random.random() > 0.5:
                result_img[x, y, j] = 255

    return result_img


def papper_noise(src_img, percent):
    a, b, c = src_img.shape
    result_img = np.copy(src_img)
    Z = int(a * b * percent)
    for i in range(Z):
        for j in range(c):
            x = random.randint(0, a - 1)
            y = random.randint(0, b - 1)
            if random.random() < 0.5:
                result_img[x, y, j] = 0                         #                    #import cv2 as cv
                                                                #                    #import numpy as np
    return result_img                                           #                    #from PIL import Image
                                                                #                    #from skimage import util
img = cv2.imread("lenna.png")                                   #                     #img = cv.imread("lenna.png")
img2=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)                        #或者可以直接调用封装好的函数 noise_gs_img=util.random_noise(img,mode='localvar')
                                                                                        #cv.imshow("source",img)
noiseImg = saltPapper_noise(img2,0.8)                                                   #cv.imshow("lenna",noise_gs_img)
noiseImg2 = papper_noise(img2,0.8)
noiseImg3 = salt_noise(img2,0.8)                                                        #cv.waitKey(0)
                                                                                        #cv.destoryAllWindows()
plt.show()
plt.figure()                                                                        #直接调用是很方便但

ax=plt.subplot(221)
plt.imshow(img2)
plt.title("src img")

ax=plt.subplot(222)
plt.imshow(noiseImg)
plt.title("papper&salt noise img")

ax=plt.subplot(223)
plt.imshow(noiseImg2)
plt.title("papper noise img")

ax=plt.subplot(224)
plt.imshow(noiseImg3)
plt.title("salt noise img")
plt.show()