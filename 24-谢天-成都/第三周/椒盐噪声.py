import cv2
import numpy as np
import random


def saltpepper_noise(img,percentage):
    h,w=img.shape[:2]
    img_n,img_sum=np.copy(img),int(h*w*percentage)

    for i in range(img_sum):
        x_r,y_r=random.randint(0,h-1),random.randint(0,w-1)
        p=random.random()
        #设置边界
        if(p<=0.5):
            img_n[x_r,y_r]=0
        else:
            img_n[x_r,y_r]=255
    return img_n

if __name__=="__main__":
    img=cv2.imread('C:\\Users\\32496\\Desktop\\lenna.png')
    img_g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gauss=saltpepper_noise(img_g,0.1)
    cv2.imshow("gray",img_g)
    cv2.imshow("saltpepper",img_gauss)
    cv2.waitKey(0)
