from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#实现灰度化
def image_gray(img):
    h,w=img.shape[:2]
    img_g=np.zeros([h,w],img.dtype)
    for i in range(h):
        for j in range(w):
            x = img[i,j]
            img_g[i,j]=int(x[2]*0.3+x[1]*0.59+x[0]*0.11)
    return img_g

#实现二值化
def img_binary(img):
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if (img[i,j]<=0.5):
                img[i,j]=0
            else:
                img[i,j]=1
    return img

if __name__=='__main__':
    img = cv2.imread("C:\\Users\\32496\\Desktop\\lenna.png")
    print("image lenna:")
    print(img)

    #灰度化
    img_g=image_gray(img)
    #img_g=cv2.cvtColor(img,COLOR_BGR2GRAY) 函数调用
    #img_g=rgb2gray(img) 函数调用
    print("img_gray:-----"%img_g)
    cv2.imshow("image show gray",img_g)

    plt.subplot(221)
    img = plt.imread("C:\\Users\\32496\\Desktop\\lenna.png")
    plt.imshow(img)

    plt.subplot(222)
    plt.imshow(img_g,cmap="gray")


    #二值化
    img_g=rgb2gray(img)
    img_b=img_binary(img_g)
    # img_b=np.where(img_g>=0.5,1,0) 函数调用
    print("img_binary:-----"%img_b)
    plt.subplot(223)
    plt.imshow(img_b,cmap="gray")
    plt.show()
