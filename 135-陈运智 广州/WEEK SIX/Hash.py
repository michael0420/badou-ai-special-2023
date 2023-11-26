import cv2

# 均值哈希
def ahash(img):
    # 缩放图片
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    # 图像转换为灰度图
    gray_img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 设置像素均值变量s和哈希值变量hash_str初始值
    s=0
    hash_str=''
    for i in range(8):
        for j in range(8):
            s = s+gray_img[i,j]
    # 求灰度像素均值
    avg=s/(8*8)
    for i in range(8):
        for j in range(8):
            if gray_img[i,j]>avg:
                hash_str =hash_str+'1'
            else:
                hash_str =hash_str+'0'
    return hash_str
# 差值哈希
def bhash(img):
    # 图片缩放
    img =cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    #图片转灰度图
    gray_img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 图片转换为哈希值
    hash_str =''
    for i in range(8):
        for j in range(8):
            if gray_img[i,j]>gray_img[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return  hash_str

def cmpHash(hash1,hash2):
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return n
img1=cv2.imread("lenna.png")
img2=cv2.imread("lenna_noise.png")
i=ahash(img1)
i1=ahash(img2)
n=cmpHash(i,i1)
j=bhash(img1)
j1=bhash(img2)
n1=cmpHash(j,j1)
print(i)
print(i1)
print(n)
print(j)
print(j1)
print(n1)
