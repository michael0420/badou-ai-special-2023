import cv2 as cv
import numpy as np


def img2Gray(filePath):
    image = cv.imread(filePath)# 读原图 ，opencv默认是BGR
    height,width = image.shape[:2]
    gray = np.zeros((height, width), dtype=np.uint8) #必须用uint8
    for i in range(height):
        for j in range(width):
            # 使用灰度转换公式
            gray[i,j ] = np.dot( image[i,j,:] , np.array([[0.11],[0.59],[0.3]]))
    cv.imshow("gray",gray)
    return gray



def BinaryImage(gray, thresh_val):
    tmp = np.where((gray <= thresh_val), gray, 255)
    ans = np.where((tmp>thresh_val), tmp, 0)
    cv.imshow("binaryImage",ans)


if __name__ == "__main__":
    BinaryImage(img2Gray("test.jpg"),200)

    cv.waitKey()
