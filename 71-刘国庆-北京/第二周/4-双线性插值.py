# 导入必要的库
import cv2
import numpy as np

# 定义双线性插值函数
def bilinear(img, outImg):
    # 获取原始图像的高度、宽度和通道数
    img_h, img_w, img_c = img.shape
    # 获取目标图像的高度和宽度
    out_h, out_w = outImg[1], outImg[0]
    # 输出原始图像和目标图像的尺寸
    print("img_h, img_w = ", img_h, img_w)
    print("out_h, out_w = ", out_h, out_w)
    # 如果原始图像的尺寸与目标尺寸相同，直接返回原始图像
    if img_h == out_h and img_w == out_w:
        return img.copy()
    # 创建一个全零矩阵，用于存储插值后的图像
    emptyImage = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    # 计算原始图像到目标图像的缩放比例
    scale_x, scale_y = float(img_w) / out_w, float(img_h) / out_h
    # 对图像的每个通道进行插值操作
    for i in range(3):
        for out_y in range(out_h):
            for out_x in range(out_w):
                # 计算目标图像中当前像素在原始图像中的坐标（使用几何中心对称法）
                img_x = (out_x + 0.5) * scale_x - 0.5
                img_y = (out_y + 0.5) * scale_y - 0.5
                # 找到用于插值计算的四个最近邻像素的坐标
                img_x0 = int(np.floor(img_x))
                img_x1 = min(img_x0 + 1, img_w - 1)
                img_y0 = int(np.floor(img_y))
                img_y1 = min(img_y0 + 1, img_h - 1)
                # 计算插值结果
                temp0 = (img_x1 - img_x) * img[img_y0, img_x0, i] + (img_x - img_x0) * img[img_y0, img_x1, i]
                temp1 = (img_x1 - img_x) * img[img_y1, img_x0, i] + (img_x - img_x0) * img[img_y1, img_x1, i]
                emptyImage[out_y, out_x, i] = int((img_y1 - img_y) * temp0 + (img_y - img_y0) * temp1)
    # 返回插值后的图像
    return emptyImage


if __name__ == '__main__':
    # 读取名为'lenna.png'的图像
    img = cv2.imread('lenna.png')
    # 调用bilinear_interpolation函数，将图像缩放为(700, 700)的尺寸
    dst = bilinear(img, (700, 700))
    # 显示双线性插值后的图像
    cv2.imshow('bilinear interp', dst)
    # 等待用户按下任意键，然后关闭窗口
    cv2.waitKey()
