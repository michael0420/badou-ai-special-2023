import cv2
import numpy as np
def function(img):
    height,width,channels = img.shape
    emptyImage=np.zeros((1920,1920,channels),np.uint8)
    sh = 1920/height
    sw = 1920/width
    for i in range(1920):
        for j in range(1920):
            x = int(i/sh + 0.5) #int()转为整形，+0.5是为了向下取整
            y = int(j/sw + 0.5)
            emptyImage[i,j]=img[x,y]
    return emptyImage
img = cv2.imread("123.png")
zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearrest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey()