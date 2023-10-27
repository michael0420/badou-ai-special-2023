import cv2
import numpy as np

# 图像灰度化
# 读取图片基础信息
src_img = cv2.imread('lenna.png')
h, w, channels = src_img.shape
# 创建等大空白图
dst_img = np.zeros([h, w, channels], src_img.dtype)
# 为空白图赋灰度值
for i in range(h):
    for j in range(w):
        # 获取原图的对应像素点
        m = src_img[i, j]
        dst_img[i, j] = m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3  # gbr

#
# # 编码显示为中文
# def zh_cn(string):
#     return string.encode('gb2312').decode(errors='ignore')


# 输出图片
# cv2.imshow(zh_cn("手动灰度化"), dst_img)
cv2.imshow("手动灰度化", dst_img)
cv2.waitKey(0)

