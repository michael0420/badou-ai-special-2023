import  cv2
import numpy as np


def function(img):
    height,width,channels = img.shape
    emptyImage = np.zeros((800,800,channels),np.uint8)
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            emptyImage[i,j] = img[x,y]
    return emptyImage


img = cv2.imread('lenna.png')
imgtool = cv2.resize(img,(800,800),200)
zoom = function(img)

print(zoom)
print(zoom.shape)
cv2.imshow("asd",zoom)
cv2.imshow("imng",img)
cv2.imshow("imgtool",imgtool)

cv2.waitKey(0)