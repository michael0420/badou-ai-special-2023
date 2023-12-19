import cv2
import numpy as np

#利用邻近插值法生成缩放图片，输入图片高度和宽度
def function(img,h,w):
    height,width,channels=img.shape
    emptyImage = np.zeros((h,w,channels),np.uint8)#新建一个800*800的图片
    # 与原图大小对比
    sh=h/height
    sw=w/width
    # 给空图片的每个像素赋值
    for i in range(h):
        for j in range(w):
            x=int(i/sh+0.5)#找第i行位置上原图相近的点
            y=int(j/sw+0.5)#找第j列位置上原图相近的点
            emptyImage[i,j]=img[x,y]#把原图x,y的点赋值给新图片的i,j位置
    return emptyImage
img=cv2.imread("lenna.png")
zoom = function(img,800,1000)
print(zoom)
print(zoom.shape)
cv2.imwrite("nearest.png",zoom)
