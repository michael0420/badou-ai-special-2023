import cv2 as cv
import numpy as np


def resize(image, output):
    originHeight, originWidth, originChannels = image.shape
    outputHeight, outputWidth, outputChannels = output.shape
    xScale, yScale = outputHeight / originHeight, outputWidth / originWidth
    for i in range(outputHeight):
        for j in range(outputWidth):
            # 先找映射的位置
            srcX, srcY = (i + 0.5) / xScale - 0.5, (j + 0.5) / xScale - 0.5
            # 再找邻近的四个点（x0,y0） (x1,y0) (x0,y1) (x1,y1)

            x0 = int(np.floor(srcX))  # 向下取整
            x1 = int(np.floor(srcX + 0.5))  # 向上取整
            y0 = int(srcY)  # 向下取整
            y1 = int(srcY + 0.5)  # 向上取整

            # 再套双线性插值的公式
            output[i, j] = (image[x0, y0] * (x1 - srcX) * (y1 - srcY) +
                            image[x1, y0] * (srcX - x0) * (y1 - srcY) +
                            image[x0, y1] * (x1 - srcX) * (srcY - y0) +
                            image[x1, y1] * (srcX - x0) * (srcY - y0)
                            )
    return output


if __name__ == "__main__":
    image = cv.imread("test.jpg")
    height, width, channels = image.shape
    output = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
    output = resize(image, output)
    cv.imshow("after resized image", output)
    cv.waitKey()
