from skimage.color import rgb2gray
import  numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
import cv2

img=cv2.imread("lenna.png")
h,w=img.shape[0:2]
#灰度化
img_gray=np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m=img[i,j]
        img_gray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)

#print(img_gray)
cv2.imshow("gray",img_gray)

# plt.subplot(220)
# img=plt.imread("lenna.png")
# plt.imshow(img)
# plt.show()
#二值化
img_gray = rgb2gray(img)
cv2.imshow("123",img_gray)
rows,cols=img_gray.shape
for i in range(rows):
    for j in range(cols):
        if(img_gray[i,j]<=0.5):
            img_gray[i,j]=0
        else:
            img_gray[i,j]=1
cv2.imshow("2",img_gray)
cv2.waitKey()
