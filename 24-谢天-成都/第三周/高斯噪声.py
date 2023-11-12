import cv2
import numpy as np
import random


def gauss_noise_generator(img,mean,sigma,percentage):

    h,w=img.shape[:2]
    img_n,img_sum=np.copy(img),int(h*w*percentage)

    for i in range(img_sum):
        x_r,y_r=random.randint(0,h-1),random.randint(0,w-1)
        img_n[x_r,y_r]+=random.gauss(mean,sigma)
        #设置边界
        if(img_n[x_r,y_r]<0):
            img_n[x_r,y_r]=0
        if (img_n[x_r, y_r]>255):
            img_n[x_r,y_r]=255
    return img_n

if __name__=="__main__":
    img=cv2.imread('C:\\Users\\32496\\Desktop\\lenna.png')
    img_g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gauss=gauss_noise_generator(img_g,0,2,1)
    cv2.imshow("gray",img_g)
    cv2.imshow("gauss",img_gauss)
    cv2.waitKey(0)
