import cv2
import numpy as np

# 均值哈希算法
def avg_hash(filename):
    '''
    1.缩放8*8
    2.灰度化
    3.灰度图像素求均值
    4.像素值大于平均值记为1，否则记为0
    5.组合起来作为图像指纹
    6.两幅图对比指纹，不同位置数越少，图像相似度越高，一般8个以下认为是同一张图
    '''
    img = cv2.imread(filename)
    # 缩放
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    # 转灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 求均值
    mean = np.mean(gray)
    # 计算指纹
    hash_str=''
    for i in range(8):
        for j in range(8):
            if gray[i,j] > mean:
                hash_str += '1'
            else:
                hash_str += '0'
    
    return hash_str

# 对比哈希值 
def cmp_hash(hash1,hash2):
    if len(hash1) != len(hash2):
        return -1
        
    n = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return n


def dec_hash(filename):
    '''
    1.缩放8*9
    2.灰度化
    3.遍历行，当前行每个像素点和后一个像素点作差，大于记为1，否则记为0，得到8*8图像指纹
    4.组合起来作为图像指纹
    5.两幅图对比指纹，不同位置数越少，图像相似度越高，一般8个以下认为是同一张图
    '''
    img = cv2.imread(filename)
    # 缩放
    img = cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    
    # 转灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   
    # 计算指纹
    hash_str = ''
    for i in range(8):
        for j in range(8):
            
            if gray[i,j] > gray[i,j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    
    return hash_str
            


   
hash_str1 = avg_hash('lenna.png')
print(hash_str1)

hash_str2 = avg_hash('lenna_noise.png')
print(hash_str2)

# 对比哈希值
n = cmp_hash(hash_str1,hash_str2)
print(n)



# 差值哈希
hash_str1 = dec_hash('lenna.png')
print(hash_str1)

hash_str2 = dec_hash('lenna_noise.png')
print(hash_str2)

# 对比哈希值
n = cmp_hash(hash_str1,hash_str2)