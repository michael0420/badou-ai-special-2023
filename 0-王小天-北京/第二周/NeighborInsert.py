import cv2
import numpy as np
from matplotlib import pyplot as plt

# 最邻近插值算法
def neighbor_insert(img, dst_h, dst_w):
    #获取原始图像像素大小
    src_h, src_w, channels = img.shape
    
    #创建最近邻插值的图像。指定长、宽，空的图像
    insert_img = np.zeros((dst_h, dst_w, channels), img.dtype)

    #如果长或宽比例缩小，返回空图像
    if dst_h < src_h or dst_w < src_w:
        return None

    #获取缩放比例
    h_ratio = dst_h / src_h
    w_ratio = dst_w / src_w
    
    #遍历新空图像，通过缩放比例，获取对应点相应的原始图像像素。
    for i in range(dst_h):
        for j in range(dst_w):
            x = int(i / h_ratio)
            y = int(j / w_ratio)

            #赋值新空图像对应原始图像的像素

            insert_img[i, j] = img[x, y]

    return insert_img

#测试图像
imge=cv2.imread("lenna.jpg")

#最近邻插入后的图像
insert_img=neighbor_insert(imge,1200, 1200)

#输出插入后图像的大小
print(insert_img.shape)

#插入
cv2.imwrite("insert_img.jpg",insert_img)
