from PIL import Image
import numpy as np
import cv2


#灰度化和二值化练习


#************封装的函数方法实现*********************

# 打开图像
image = Image.open('lenna.png')

# 灰度化
gray_image = image.convert('L')

# 保存灰度图像
gray_image.save('gray_lenna.png')

# 二值化（使用固定阈值）
threshold = 128
binary_image = gray_image.point(lambda x: 0 if x < threshold else 255, '1')

# 保存二值图像
binary_image.save('2rgb_lenna.png')

#************封装的函数方法实现*********************



#************普通方法实现*********************

# 打开图像
image = Image.open('lenna.png')

# 灰度化（按照公式计算）
gray_image = image.convert('L')
# 获取图像的宽度和高度
width, height = gray_image.size
# 遍历图像的每一个像素点，根据公式计算灰度值
for x in range(width):
    for y in range(height):
        r, g, b = gray_image.getpixel((x, y))
        gray = int(r * 0.299 + g * 0.587 + b * 0.114)
        # 设置灰度值
        gray_image.putpixel((x, y), (gray, gray, gray))

# 保存灰度图像
gray_image.save('gray_image.jpg')

# 二值化（使用计算阈值）
# 计算图像的平均灰度值
total_gray = 0
for x in range(width):
    for y in range(height):
        total_gray += gray_image.getpixel((x, y))[0]
avg_gray = total_gray / (width * height)

# 根据平均灰度值计算阈值，对图像进行二值化处理
binary_image = gray_image.point(lambda x: 0 if x < avg_gray else 255, '1')

# 保存二值图像
binary_image.save('binary_image.jpg')


#************封装的函数方法实现*********************