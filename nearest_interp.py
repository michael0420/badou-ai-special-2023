import cv2
import numpy as np
#最邻近插值
def nearest_inter(img):
    hw = 500
    h, w, channels =img.shape
    emptyImg = np.zeros((hw,hw,channels), np.uint8)
    sh = hw/h
    sw = hw/w

    for i in range(hw):
        for j in range(hw):
            emptyImg[i, j]=img[int(i/sh+0.5),int(j/sw+0.5)]
    return emptyImg
img = cv2.imread("lenna.png")
zoom = nearest_inter(img)
print(zoom)
print(zoom.shape)
cv2.imshow("sss",zoom)
# cv2.imshow("",img)
cv2.waitKey(0)