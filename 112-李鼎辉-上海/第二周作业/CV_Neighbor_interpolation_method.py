import cv2
import numpy as np
from matplotlib import pyplot as plt

# 最邻近插值算法
# def img2Neighbor(img):
#     srcHeight, srcWidth, srcChannels = img.shape
#     # dstHeight = srcHeight*2
#     # dstWidth = srcWidth*2
#     CreateImage = np.zeros((600, 600, srcChannels), np.uint8)
#     dstH = 600/srcHeight
#     dstW = 600/srcWidth
#     for i in range(600):
#         for j in range(600):
#             x = int(i/dstW+0.5)
#             y = int(j/dstH + 0.5)
#             CreateImage[i,j]=img[x,y]
#     return CreateImage

def img2Neighbor(img, dstHeight, dstWidth):
    srcHeight, srcWidth, srcChannels = img.shape
    CreateImage = np.zeros((dstHeight, dstWidth, srcChannels), np.uint8)
    dstH = dstHeight / srcHeight
    dstW = dstWidth / srcWidth
    for i in range(dstHeight):
        for j in range(dstWidth):
            x = int(i / dstH + 0.5)
            y = int(j / dstW + 0.5)
            # 确保 x 和 y 在合理范围内
            x = max(0, min(x, srcHeight - 1))
            y = max(0, min(y, srcWidth - 1))
            CreateImage[i, j] = img[x, y]
    return CreateImage


imge=cv2.imread("lenna.png")
imgNeighbor=img2Neighbor(imge,800, 800)
print(imgNeighbor)
print(imgNeighbor.shape)
cv2.imshow("nearest_interp", imgNeighbor)
cv2.imshow("image", imge)
cv2.waitKey(0)
