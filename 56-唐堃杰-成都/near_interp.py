import cv2
import numpy


def get_near_interp(cur_img):
    src_h, src_w,channels = img.shape
    dst_h = 800
    dst_w = 800
    new_img = numpy.zeros([dst_h,dst_w,channels], cur_img.dtype)
    h_ratio = dst_h / src_h
    w_ratio = dst_w / src_w
    for i in range(dst_h):
        for j in range(dst_w):
            # 这里加0.5是因为int向下取整，此时这里为了四舍五入
            x = int((i / h_ratio) + 0.5)
            y = int((j / w_ratio) + 0.5)
            new_img[i, j] = img[x, y]
    return new_img


if __name__ == "__main__":
    img = cv2.imread("lenna.png")
    data = get_near_interp(img)
    print(data)
    cv2.imshow("near interp", data)
    cv2.imshow("image", img)
    # 这里是挂起，为了防止直接结束
    cv2.waitKey(0)