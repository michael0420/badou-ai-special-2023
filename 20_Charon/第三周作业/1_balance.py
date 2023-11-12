import cv2
import numpy as np

path = r"C:\Users\Administrator\Desktop\homework\lenna.png"
padding = 2 # 填充的像素圈数
size = 3    # 卷积核大小

# 均衡函数
def Balance(pad_img, i, j, size):
    sum = 0
    for x in range(size):
        for y in range(size):
            sum += pad_img[i + x, j + y]
    return round(sum / (size * size))

# 读取原图获取基础信息
src_img = cv2.imread(path, 0)
h, w = src_img.shape[:2]
# 输出图
dst_img = np.zeros([h, w] , src_img.dtype)
# 填充边框
pad_img = np.zeros([h + padding * 2, w + padding * 2], src_img.dtype)
pad_img[padding : h+padding, padding : w+padding] = src_img
# 循环为输出图的每个像素点赋值
for i in range(w):
    for j in range(h):
        dst_img[i, j] = Balance(pad_img, i, j, size)


blur = cv2.blur(src_img, (3, 3))

cv2.imshow('dst_img', dst_img)
cv2.imshow('blur', blur)
cv2.waitKey(0)







