import cv2
import numpy as np

def function(img):
   h,w,c = img.shape
   newImage = np.zeros((1000,1000,c),np.uint8)
   sh = 1000/h
   sw = 1000/w
   for i in range(1000):
       for j in range(1000):
           x=int(i/sh + 0.5)
           y=int(j/sw + 0.5)
           newImage[i,j] = img[x,y]
   return newImage

if __name__ == '__main__':
    img= cv2.imread("lenna.png")
    newImage = function(img)
    cv2.imshow("nearest",newImage)
    cv2.imshow("image",img)
    cv2.waitKey()