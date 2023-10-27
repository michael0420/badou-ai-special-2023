"""
2.实现最邻近插值
"""

"""
清晰度不会特别高。原理：
1. x/size = i/new_size  -> i = x * new_size / size
2. 得到的结果是一个浮点数，因为i为位置，是整数
3. 实现四舍五入 int(x + 0.5) x是浮点数
"""
import cv2
import numpy as np


# 最邻近插值
# ----------------------原理-----------------------------#
def function(img):
    height, width, channels = img.shape
    size = 800
    emptyImage = np.zeros((size, size, channels), np.uint8)
    for i in range(size):
        for j in range(size):
            x = int(i * height / size + 0.5)  # int(),转为整型，使用向下取整。 +0.5是为了实现四舍五入
            y = int(j * width / size + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage


img = cv2.imread("lenna.png")
zoom = function(img)
# ----------------------原理-----------------------------#

# ----------------------调包-----------------------------#
# img = cv2.imread("lenna.png")
# height, width, channels = img.shape
# size = 800
# img = cv2.resize(img, (size, size))
# ----------------------调包-----------------------------#


print("-----img------")
print(img)
print(img.shape)

print("-----zoom------")
print(zoom)
print(zoom.shape)

cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)
