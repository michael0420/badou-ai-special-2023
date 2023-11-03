from PIL import Image

#双线性插值练习


# 打开原始图像
original_image = Image.open('lenna.png')

# 缩放比例
scale = 2

# 计算缩放后的图像尺寸
width, height = original_image.size
new_width = int(width / scale)
new_height = int(height / scale)

# 创建目标图像
target_image = Image.new('RGB', (new_width, new_height))

# 双线性插值
for x in range(new_width):
    for y in range(new_height):
        # 计算源图像中四个最近邻点的位置
        source_x1 = int(x * scale)
        source_y1 = int(y * scale)
        source_x2 = min(source_x1 + 1, width - 1)
        source_y2 = min(source_y1 + 1, height - 1)
        # 计算在模型下的位置，u,v 是归一化的，最终实际的位置要乘上原图尺寸，\
        #再减去0.5是为了防止计算定位过多，而导致超界
        u = x * scale - source_x1
        v = y * scale - source_y1
        # 计算在四个最近邻点的像素值
        color1 = original_image.getpixel((source_x1, source_y1))
        color2 = original_image.getpixel((source_x1, source_y2))
        color3 = original_image.getpixel((source_x2, source_y1))
        color4 = original_image.getpixel((source_x2, source_y2))
        # 使用双线性插值计算当前像素点的颜色值
        red = int((1 - u) * (1 - v) * color1[0] + u * (1 - v) * color3[0] \
            + (1 - u) * v * color2[0] + u * v * color4[0])
        green = int((1 - u) * (1 - v) * color1[1] + u * (1 - v) * color3[1] \
            + (1 - u) * v * color2[1] + u * v * color4[1])
        blue = int((1 - u) * (1 - v) * color1[2] + u * (1 - v) * color3[2] \
            + (1 - u) * v * color2[2] + u * v * color4[2])
        # 将计算得到的颜色值设置为目标图像当前像素点的颜色
        target_image.putpixel((x, y), (red, green, blue))

# 保存目标图像
target_image.save('image.jpg')