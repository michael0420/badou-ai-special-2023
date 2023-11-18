import numpy as np
import cv2


def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # 在控制台打印鼠标当前位置的坐标
        print(f"当前坐标点：({x}, {y})")


if __name__ == '__main__':

    # 读取图片
    img = cv2.imread("C:/Users/15082/Desktop/10.jpg")

    # 创建图像窗口并设置鼠标回调函数
    # cv2.namedWindow('Original Image')
    # cv2.setMouseCallback('Original Image', show_coordinates)

    # 显示图像，等待用户按下ESC键退出
    # while True:
    #     cv2.imshow('Original Image', img)
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == 27:  # 27是ESC键的ASCII码
    #         break

    # cv2.destroyAllWindows()

    # 定义透视变换的原点和目标点
    src_points = np.float32([[49, 56], [129, 24], [121, 176], [220, 117]])
    dst_points = np.float32([[0, 0], [300, 0], [0, 400], [300, 400]])

    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 进行透视变换
    warp_img = cv2.warpPerspective(img, perspective_matrix, (300, 400))

    # 显示透视变换前的图片和透视变换后的图片
    cv2.imshow('origin_img', img)
    cv2.imshow('warp_img', warp_img)
    cv2.waitKey(0)
