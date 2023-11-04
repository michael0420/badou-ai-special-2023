
import cv2 as cv
import numpy as np


def resize(image, newHeight, newWidth):
    height, width ,channel= image.shape
    ans = np.zeros((newHeight, newWidth, 3) , dtype=np.uint8)
    heightRatio, widthRatio = newHeight/height, newWidth/width

    for i in range(newHeight):
        for j in range(newWidth):

            sourceI,sourceJ = round(i/heightRatio), round(j/widthRatio)
            if sourceI >= height:
                sourceI = height-1
            if sourceJ >=width:
                sourceJ = width-1
            ans[i,j] = image[sourceI,sourceJ]
    cv.imshow("nearest", ans)

if __name__ == "__main__":
    image = cv.imread("test.jpg")
    height,width,channel = image.shape
    height,width = height*3, width*3
    resize(image, height,width )
    cv.waitKey()