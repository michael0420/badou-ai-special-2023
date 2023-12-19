# 导入OpenCV库
import cv2


# 定义均值哈希算法函数
def aHash(img):
    # 缩放为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s += img_gray[i, j]
    # 求平均灰度
    avg = s / 64
    # 灰度大于平均值为1,相反为0,生成图片的hash值
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'
    # 返回哈希值
    return hash_str


# 定义差值哈希算法函数
def dHash(img):
    # 缩放8*9
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # hash_str为hash值初值为''
    hash_str = ''
    # 每行前一个像素大于后一个像素,hash值为1，相反,hash值为0，生成哈希
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > img_gray[i, j + 1]:
                hash_str += '1'
            else:
                hash_str += '0'
    # 返回哈希值
    return hash_str


# 哈希值对比函数
def cmpHash(hash_1, hash_2):
    # 相似度初始值为0
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash_1) != len(hash_2):
        return -1
    # 遍历判断,不相等则n计数+1，n最终为相似度
    for i in range(len(hash_1)):
        if hash_1[i] != hash_2[i]:
            n += 1
    # 返回相似度n
    return n


# 读取图像1
img1 = cv2.imread("lenna.png")
# 读取图像2
img2 = cv2.imread("lenna_noise.png")
# 打印输出图像1和图像2的均值哈希值
print(f"图像1的均值哈希值:{aHash(img1)}\n图像1的均值哈希值:{aHash(img2)}")
# 比较两个哈希值的相似度并打印输出
print(f"两个哈希值的相似度{cmpHash(aHash(img1), aHash(img2))}")
# 打印输出图像1和图像2的差值哈希值
print(f"图像1的均值哈希值:{dHash(img1)}\n图像1的均值哈希值:{dHash(img2)}")
# 比较两个哈希值的相似度并打印输出
print(f"两个哈希值的相似度{cmpHash(dHash(img1), dHash(img2))}")
