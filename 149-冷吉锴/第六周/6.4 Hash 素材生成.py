import cv2
import numpy as np
import os.path as path
from PIL import Image
from PIL import ImageEnhance


# 修改图片的亮度
def light(image):
    return np.uint8(np.clip((1.3 * image + 10), 0, 255))


# 缩放图片
def resize(image):
    return cv2.resize(image, (0, 0), fx=1.25, fy=1)


# 修改图片的对比度
def contrast(image):
    def contrast_brightness_image(src1, a, g):
        """
        粗略的调节对比度和亮度
        @param src1: 图片
        @param a: 对比度
        @param g: 亮度
        @return:
        """
        # 获取shape的数值，height和width和通道
        h, w, ch = src1.shape

        # 新建全零图片数组src2，将height和width类型设置为原图片的通道类型（色素全为零，输出为全黑图片）
        src2 = np.zeros([h, w, ch], src1.dtype)
        # addWeighted 函数说明如下
        return cv2.addWeighted(src1, a, src2, 1 - a, g)
    return contrast_brightness_image(image, 1.2, 1)


# 锐化操作
def sharp(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    return cv2.filter2D(image, -1, kernel=kernel)


# 模糊操作
def blur(image):
    return cv2.blur(image, (15, 1))


# 色度增强
def enhance_color(image):
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    return enh_col.enhance(color)


# 旋转图片
def rotate(image):
    def rotate_bound(image, angle):
        # 抓取图像的尺寸，然后确定中心点
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # 抓取旋转矩阵（应用角度的负数顺时针旋转），然后抓取正弦和余弦（即矩阵的旋转分量）
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # 计算图像的新边界尺寸
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # 调整旋转矩阵以考虑平移
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # 执行实际旋转并返回图像
        return cv2.warpAffine(image, M, (nW, nH))
    return rotate_bound(image, 45)


# 保存图片
def save_img(image, img_name, output_path=None):
    cv2.imwrite(path.join(output_path, img_name), image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    pass


def main():
    data_img_name = 'lenna.png'
    output_path = './source'
    data_path = path.join(output_path, data_img_name)

    img = cv2.imread(data_path)

    # 修改图片的亮度
    img_light = light(img)
    # 修改图片的大小
    img_resize = resize(img)
    # 修改图片的对比度
    img_contrast = contrast(img)
    # 锐化
    img_sharp = sharp(img)
    # 模糊
    img_blur = blur(img)
    # 色度增强
    img_color = enhance_color(Image.open(data_path))
    # 旋转
    img_rotate = rotate(img)
    img_rotate1 = Image.open(data_path).rotate(45)
    # 两张图片横向合并（便于对比显示）
    # tmp = np.hstack((img, img_rotate))

    save_img(img_light, "%s_light.jpg" % data_img_name.split(".")[0], output_path)
    save_img(img_resize, "%s_resize.jpg" % data_img_name.split(".")[0], output_path)
    save_img(img_contrast, "%s_contrast.jpg" % data_img_name.split(".")[0], output_path)
    save_img(img_sharp, "%s_sharp.jpg" % data_img_name.split(".")[0], output_path)
    save_img(img_blur, "%s_blur.jpg" % data_img_name.split(".")[0], output_path)
    save_img(img_rotate, "%s_rotate.jpg" % data_img_name.split(".")[0], output_path)
    # 色度增强
    img_color.save(path.join(output_path, "%s_color.jpg" % data_img_name.split(".")[0]))
    img_rotate1.save(path.join(output_path, "%s_rotate1.jpg" % data_img_name.split(".")[0]))


if __name__ == '__main__':
    main()
