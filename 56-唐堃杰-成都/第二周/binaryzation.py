import cv2
import numpy
from skimage.color import rgb2gray

def parse_binary(cur_img):
    row, col = cur_img.shape[:2]
    print(row, col)
    for i in range(row):
        for j in range(col):
            if cur_img[i, j] <= 0.5:
                cur_img[i, j] = 0
            else:
                cur_img[i, j] = 1
    new_img = numpy.where(cur_img >= 0.5, 1, 0)
    print(new_img)


if __name__ == "__main__":
    # 读取文件信息，以BGR的方式
    img = cv2.imread("lenna.png")
    img_gray = rgb2gray(img)
    print(type(img), type(img_gray))
    parse_binary(img_gray)
