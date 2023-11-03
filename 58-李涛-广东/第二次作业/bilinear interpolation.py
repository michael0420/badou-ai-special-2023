import numpy as np
import cv2
 
'''
双线性插值的Python实现
功能：上采样使用双线性插值法
python implementation of bilinear interpolation
''' 
def bilinear_interpolation(img, out_dim):  # 参数：img（输入图像）和out_dim（输出图像的尺寸）
    src_h, src_w, channel = img.shape  # 获取输入图像的高度、宽度和通道数
    dst_h, dst_w = out_dim[1], out_dim[0]  # 获取输出图像的高度和宽度
    print("src_h, src_w = ", src_h, src_w)  
    print("dst_h, dst_w = ", dst_h, dst_w) 
    if src_h == dst_h and src_w == dst_w:  # 如果输入图像的尺寸与输出图像的尺寸相同，则直接返回输入图像的副本
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)  # 创建一个全零的三维数组，用于存储输出图像的像素值
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h  # 计算水平和垂直方向上的缩放比例
    for i in range(3):  # 遍历RGB三个通道
        for dst_y in range(dst_h):  # 遍历输出图像的每一行
            for dst_x in range(dst_w):  # 遍历输出图像的每一列
                # find the origin x and y coordinates of dst image x and y 求DST图像x和y的原点x和y坐标
                # use geometric center symmetry 使用几何中心对称
                src_x = (dst_x + 0.5) * scale_x - 0.5  # 根据缩放比例和当前像素位置计算源图像中的x坐标
                src_y = (dst_y + 0.5) * scale_y - 0.5  # 根据缩放比例和当前像素位置计算源图像中的y坐标
                # find the coordinates of the points which will be used to compute the interpolation 
                # 找出将用于计算插值的点的坐标
                src_x0 = int(np.floor(src_x))  # 向下取整，得到源图像中x坐标的整数部分
                src_x1 = min(src_x0 + 1, src_w - 1)  # 确保源图像中x坐标的整数部分不超过源图像的宽度减一
                src_y0 = int(np.floor(src_y))  # 向下取整，得到源图像中y坐标的整数部分
                src_y1 = min(src_y0 + 1, src_h - 1)  # 确保源图像中y坐标的整数部分不超过源图像的高度减一
                # calculate the interpolation 计算插值
                # 根据插值公式计算插值结果的一部分
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]  
                # 根据插值公式计算插值结果的一部分
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]  
                # 根据插值公式计算插值结果并存储到输出图像中
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)  
                # 返回输出图像
    return dst_img  

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img,(700,700))
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey()
