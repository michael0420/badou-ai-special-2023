import cv2 as cv

def zh_ch(string):
	return string.encode("gbk").decode(errors="ignore")
def equalization(origin_image):
    height, width = origin_image.shape
    histogram = {}
    # 直方图初始化为0
    for i in range(256):
        histogram.update({i: 0})
    # 计算直方图
    for i in range(height):
        for j in range(width):
            grayVal = origin_image[i, j]
            histogram.update({grayVal: histogram.get(grayVal) + 1})
    # 计算累加直方图
    sum = {}
    sum.update({-1: 0})
    for i in range(256):
        sum.update({i:sum.get(i - 1) + histogram.get(i)})

    HW = sum.get(255)
    q = {}
    for p in range(256):
        q.update({p: int(sum.get(p) / HW * 256) - 1})

    for i in range(height):
        for j in range(width):
            grayVal = origin_image[i,j]
            origin_image[i,j] = q.get(grayVal)
    return origin_image


if __name__ == "__main__":
    image = cv.imread("test.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow(zh_ch("原始图"), image)
    output_image = equalization(image)
    officialEqualizeHist = cv.equalizeHist(image)
    cv.imshow("cv.equalizeHist", officialEqualizeHist)
    cv.imshow("my equalizeHist", output_image)

    cv.waitKey()
