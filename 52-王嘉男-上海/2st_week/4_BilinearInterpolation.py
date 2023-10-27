import numpy as np
import matplotlib.pyplot as plt


# 双线性插值函数
def bilinear_interpolation(image_source, target_size):
    height_source, width_source, channel = image_source.shape
    height_target, width_target = target_size[1], target_size[0]

    if height_source == height_target and width_source == width_target:
        return image_source.copy()
    dst_img = np.zeros((height_target, width_target, 3), dtype=np.uint8)
    x_scale, y_scale = float(width_source) / width_target, float(height_source) / height_target
    for i in range(3):
        for y_target in range(height_target):
            for x_target in range(width_target):
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if we use direct way, src_x = dst_x * scale_x
                x_source = (x_target + 0.5) * x_scale - 0.5
                y_source = (y_target + 0.5) * y_scale - 0.5

                # find the coordinates of the points which will be used to compute the interpolation
                x0_source = int(np.floor(x_source))
                x1_source = min(x0_source + 1, width_source - 1)
                y0_source = int(np.floor(y_source))
                y1_source = min(y0_source + 1, height_source - 1)

                # calculate the interpolation
                temp0 = ((x1_source - x_source) * image_source[y0_source, x0_source, i] +
                         (x_source - x0_source) * image_source[y0_source, x1_source, i])
                temp1 = ((x1_source - x_source) * image_source[y1_source, x0_source, i] +
                         (x_source - x0_source) * image_source[y1_source, x1_source, i])
                dst_img[y_target, x_target, i] = int((y1_source - y_source) * temp0 + (y_source - y0_source) * temp1)
    return dst_img


# 读取原始图
Image_source = plt.imread("hi.jpg")
# 子图位置
plt.subplot(221)
# 显示原始图
plt.imshow(Image_source)
# 获得原始图的高度和宽度
height, width = Image_source.shape[:2]
# 代入双线性插值函数
image_bilinear_interpolation = bilinear_interpolation(Image_source, ((height * 2), (width * 2)))
# 子图位置
plt.subplot(222)
# 显示双线性插值后的图片
plt.imshow(image_bilinear_interpolation)
# 图片关闭才结束运行程序
plt.waitforbuttonpress()
