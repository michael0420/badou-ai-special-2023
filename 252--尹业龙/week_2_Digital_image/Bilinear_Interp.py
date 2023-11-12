import cv2
import numpy as np

def bilinear_interp(img,out_dim):
    """
    <2023.9.8> version:1.0
    实现双线性插值算法
    :param:img,输入图片
    :param:out_dim,目标图片维度大小
    :return:empty_img,经插值处理后的图片
    """
    src_h, src_w, channel = img.shape  # 获取原图像尺寸，shape()函数中第一个参数是height，第二个参数是width
    dst_w = out_dim[0]  # 获取目标图像维度(w,h),第一个参数是width
    dst_h = out_dim[1]  # 第二个参数是height
    # 如果目标图像尺寸和原图像一致，则不需插值，直接复制原图输出
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    empty_img = np.zeros((dst_h, dst_w, channel), np.uint8)  # 新建多维数组时要遵循shape()函数，先h后w
    # 计算缩放比例
    sh = dst_h / src_h
    sw = dst_w / src_w
    # 循环处理每个通道的每个像素点
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 优化：几何中心重合---参考公式是直接乘以缩放比例的倒数，此处为乘。0.5为几何中心重合需要移动的距离，由公式计算而来
                # 这里是对原图和目标图中的坐标进行运算，当处于循环中时，相当于图像中每个像素点的坐标值都在参与相同的运算
                src_x = (dst_x + 0.5) / sw - 0.5
                src_y = (dst_y + 0.5) / sh - 0.5
                # 防溢出：坐标处理，确保计算得到的 src_x 和 src_y 不会超出边界
                src_x0 = int(np.floor(src_x))  # 取整运算
                src_x1 = min(src_x0+1, src_w-1)  # 双线性插值法就是取相邻的四个点来计算（点与点之间的距离为1）；加上src_w-1取min是为了防止超出边界
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0+1, src_h-1)
                # 像素点值计算
                tempf0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                tempf1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                empty_img[dst_y, dst_x, i] = int((src_y1 - src_y) * tempf0 + (src_y - src_y0) * tempf1)
    return empty_img




