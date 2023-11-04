from PIL import Image

#最邻近值插值练习


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

# 最邻近插值
for x in range(new_width):
    for y in range(new_height):
        # 找到源图像中最近的像素点的位置
        source_x = int(x * scale)
        source_y = int(y * scale)
        # 获取该像素点的颜色值并设置为目标图像的颜色
        color = original_image.getpixel((source_x, source_y))
        target_image.putpixel((x, y), color)

# 保存目标图像
target_image.save('image.jpg')