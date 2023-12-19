import matplotlib.pyplot as plt
import numpy as np
import cv2

plt.rcParams['font.sans-serif'] = [u'SimHei']

def Proximity_interpolation(img, H_out, W_out):
    H, W, C = img.shape
    img_output = np.zeros((H_out, W_out, C), dtype=np.uint8)
    # img_output = np.zeros((H_out, W_out, C), img.dtype)
    x_ratio, y_ratio = float(W)/W_out, float(H)/H_out
    for i in range(3):
        for y in range(H_out):
            for x in range(W_out):

                # 中心重合
                x_center = (x + 0.5) * x_ratio - 0.5
                y_center = (y + 0.5) * y_ratio - 0.5

                # 左下角坐标(x0, y0),右上角坐标(x1,y1)
                x0, y0 = int(np.floor(x_center)), int(np.floor(y_center))
                x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)

                # x_tmp1是(x0, y1)和(x1， y1)的中点值，x_tmp2是(x0, y0)和(x1， y0)的中点值
                x_tmp1 = (x_center - x0) * img[y0, x1, i] + (x1 - x_center) * img[y0, x0, i]
                x_tmp2 = (x_center - x0) * img[y1, x1, i] + (x1 - x_center) * img[y1, x0, i]

                # x_tmp1在上边， x_tmp2在下边
                img_output[y, x, i] = int((y_center - y0) * x_tmp2 + (y1 - y_center) * x_tmp1)


    return img_output

if __name__ == '__main__':
    # cv2读取图片0-255，plt读取图片0-1，需要进行转换，乘255再转化为uint8

    # cv2显示
    # img_input = cv2.imread('lenna.png')
    # img_output = Proximity_interpolation(img_input, 700, 700)
    # cv2.imshow("双线性插值图(700x700)", img_output)
    # cv2.waitKey()

    # plt显示
    img_input = plt.imread('lenna.png')
    # print(img_input)
    img_input = (img_input * 255).astype(np.uint8)
    img_output1 = Proximity_interpolation(img_input, 700, 700)

    plt.subplot(121), plt.title("原图(512x512)")
    plt.imshow(img_input)

    plt.subplot(122), plt.title("双线性插值图(700x700)")
    plt.imshow(np.uint8(img_output1))
    plt.show()
