"""
最邻近插值
"""
import cv2
import numpy as np

def function(img):
    height,width,channels = img.shape
    emptyImg = np.zeros((800,800,channels),np.uint8)
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i/sh + 0.5) #int()，转为整型，加上0.5弥补int向下取整的损失
            y = int(j/sw + 0.5)
            emptyImg[i,j] = img[x,y]
    return  emptyImg


img = cv2.imread("lenna.png")
# zoom = cv2.resize(img,(800,800),interpolation=cv2.INTER_CUBIC)

zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()