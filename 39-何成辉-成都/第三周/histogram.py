import cv2
from matplotlib import pyplot as plt

'''
第三方库的调用及参数解释：

    1、calcHist—计算图像直方图：
    函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
    images：图像矩阵，例如：[image]
    channels：通道数，例如：0
    mask：掩膜，一般为：None
    histSize：直方图大小，一般等于灰度级数
    ranges：横轴范围
    
    2、hist，它的作用是绘制直方图。这个函数的参数包括：
    x：需要绘制直方图的数据。-------------------------------------------------重要参数
    bins：直方图的柱子数量，默认为None。
    range：直方图的范围，默认为None。-----------------------------------------重要参数
    density：是否将直方图归一化，默认为False。
    weights：每个数据点的权重，默认为None。
    cumulative：是否计算累积直方图，默认为False。
    bottom：直方图的底部位置，默认为None。
    histtype：直方图的类型，默认为'bar'。
    align：柱子的对齐方式，默认为'mid'。
    orientation：直方图的方向，默认为'vertical'。
    rwidth：柱子的宽度，默认为None。
    log：是否使用对数坐标轴，默认为False。
    color：柱子的颜色，默认为None。
    label：图例的标签，默认为None。
    stacked：是否堆叠直方图，默认为False。
    data：额外的数据，默认为None。
    **kwargs：其他关键字参数。
'''


# 灰度图像直方图
# 获取灰度图像
img = cv2.imread("lenna.png", 1)
# 将图片转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("image_gray", gray)

# 灰度图像的直方图，方法一
# 创建一个新的图形窗口
plt.figure()
# 绘制灰度图像的直方图
'''
你的需求是使用matplotlib库中的plt.hist()函数来绘制灰度图像的直方图。
这个函数需要两个参数：第一个参数是要绘制的数据，第二个参数是数据的范围。
gray.ravel()作为第一个参数,256作为第二个参数。ravel()函数将二维数组转换为一维数组。
'''
plt.hist(gray.ravel(), 256)
# 显示图形
plt.show()


# 灰度图像的直方图, 方法二
'''
在使用cv2.calcHist()函数时，参数[gray]和[0]都是列表，这是因为OpenCV的函数通常接受一个或多个图像作为输入。在这个例子中，
我们只传递了一个图像（gray），所以需要将其放在一个列表中。同样，参数[0]也是一个列表，表示我们要计算直方图的通道索引。
'''
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure() # 新建一个图像
plt.title("Grayscale Histogram")
plt.xlabel("Bins") # X轴标签
plt.ylabel("# of Pixels") # Y轴标签
plt.plot(hist)
plt.xlim([0,256]) # 设置x坐标轴范围
plt.show()


# 彩色图像直方图

# 使用cv2.imread()函数读取图片文件"lenna.png"，并将结果存储在变量image中。
image = cv2.imread("lenna.png")

# 使用cv2.imshow()函数显示原始图片。
cv2.imshow("Original", image)


# cv2.waitKey(0)是OpenCV库中的一个函数，用于等待用户按键。参数0表示无限期等待，直到用户按下一个键。
# cv2.waitKey(0)

# 使用cv2.split()函数将图片的每个颜色通道分离出来，并将结果存储在变量chans中。
chans = cv2.split(image)


# 创建一个元组colors,包含三个字符串："b"、"g"和"r",分别表示蓝色、绿色和红色。
colors = ("b", "g", "r")


plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

# 使用zip()函数将chans和colors两个列表组合在一起，然后遍历每个元素。
for (chan,color) in zip(chans,colors):

    # 对于每个元素，使用cv2.calcHist()函数计算颜色直方图，并将结果存储在变量hist中。
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])

    # 使用plt.plot()函数绘制颜色直方图，设置颜色为当前颜色通道的颜色。
    plt.plot(hist,color = color)

    # 设置x轴的范围为[0,256]。
    plt.xlim([0,256])
plt.show()

'''
在OpenCV中，calcHist和hist函数都用于计算图像的直方图，但它们之间有一些区别：
    calcHist是一个更通用的函数，它可以计算任何类型的图像（如彩色、灰度等）的直方图。而hist函数主要用于计算灰度图像的直方图。
    
    calcHist函数提供了更多的参数选项，可以自定义直方图的大小、范围、归一化方法等。而hist函数则没有这么多参数选项。
    
    calcHist函数返回一个数组，其中包含了每个bin的像素值。而hist函数返回两个数组，分别表示直方图的值和bin的范围。
'''