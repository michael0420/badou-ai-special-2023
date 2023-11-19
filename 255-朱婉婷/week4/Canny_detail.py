
""""
Canny手写算法
"""

import numpy as np
import matplotlib.pyplot as plt
import math

plt.rcParams['font.sans-serif'] = ['SimHei']	# 显示中文
plt.rcParams['axes.unicode_minus'] = False		# 显示负号

pic_path = 'lenna.png'
img = plt.imread(pic_path)
if pic_path[-4:] == '.png':
    img *= 255
img = img.mean(axis=-1)#对每个像素点的RGB取均值

############1. 高斯平滑##################
sigma = 0.5
dim = int(np.round(sigma*6+1))
#确保为奇数
if dim%2 == 0:
    dim +=1
#定义高斯核
Gauss_kernel = np.zeros([dim,dim])
#计算高斯核的时候中心对称
tmp = [i-dim//2 for i in range(dim)]
#公式设置高斯核
n1 = 1/(2*sigma**2*math.pi)
n2 = -1/(2*sigma**2)
for i in range(dim):
    for j in range(dim):
        Gauss_kernel[i,j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
#高斯矩阵和为1，则变化前后图像亮度不变
Gauss_kernel = Gauss_kernel/Gauss_kernel.sum()
print("高斯核：\n",Gauss_kernel)

#平滑后图像
img_new = np.zeros(img.shape)
#边缘填补，卷积后矩阵大小不变
tmp_out = dim//2
img_pad = np.pad(img,((tmp_out,tmp_out),(tmp_out,tmp_out)),'constant')

#遍历卷积
dx = img.shape[0]
dy = img.shape[1]
for i in range(dx):
    for j in range(dy):
        img_new[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*Gauss_kernel)
plt.figure(1)
#此时img_new为浮点型数据，需要强制转化类型才可以
plt.imshow(img_new.astype(np.uint8),cmap='gray')
plt.title('Gaussian_filter')

#################2. 求梯度#################
sobel_kernel_x = np.array([[-1,0,1],
                           [-2,0,2],
                           [-1,0,1]])
sobel_kernel_y = np.array([[1,2,1],
                           [0,0,0],
                           [-1,-2,-1]])
#存储梯度图像
img_tidu_x = np.zeros(img_new.shape)
img_tidu_y = np.zeros(img_new.shape)
img_tidu = np.zeros(img_new.shape)

dim = sobel_kernel_x.shape[0]

#边缘填补，加1
img_pad = np.pad(img_new,((1,1),(1,1)),'constant')
#卷积
for i in range(dx):
    for j in range(dy):
        img_tidu_x[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*sobel_kernel_x)
        img_tidu_y[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*sobel_kernel_y)
        img_tidu[i,j] = np.sqrt(img_tidu_x[i,j]**2 + img_tidu_y[i,j]**2)

#为下一过程做准备，计算梯度
sobel_kernel_x[sobel_kernel_x==0] = 0.000001
grad = img_tidu_y/img_tidu_x

plt.figure(2)
plt.imshow(img_tidu.astype(np.uint8),cmap='gray')
plt.title('梯度图')

#######################3. 非极大值抑制####################
img_yizhi = np.zeros(img_tidu.shape)

#边缘不计算
for i in range(1,dx-1):
    for j in range(1,dy-1):
        #在8个领域内是否要抹去做标记
        flag = True
        temp = img_tidu[i-1:i+2,j-1:j+2]
        #线性插值抑制
        if grad[i,j] <= -1:
            num_1 = (temp[0,1]-temp[0,0])/grad[i,j]+temp[0,1]
            num_2 = (temp[2,1]-temp[2,2])/grad[i,j]+temp[2,1]
            #如果非最大值，则抑制
            if not (img_tidu[i,j]>num_1 and img_tidu[i,j]>num_2):
                flag = False
        elif grad[i,j] >= 1:
            num_1 = (temp[0,2]-temp[0,1])/grad[i,j]+temp[0,1]
            num_2 = (temp[2,0]-temp[2,1])/grad[i,j]+temp[2,1]
            if not (img_tidu[i,j]>num_1 and img_tidu[i,j]>num_2):
                flag = False
        elif grad[i,j] > 0:
            num_1 = (temp[0,2]-temp[1,2])*grad[i,j]+temp[1,2]
            num_2 = (temp[2,0]-temp[1,0])*grad[i,j]+temp[1,0]
            if not (img_tidu[i,j]>num_1 and img_tidu[i,j]>num_2):
                flag = False
        elif grad[i,j] < 0:
            num_1 = (temp[1,0]-temp[0,0])*grad[i,j]+temp[1,0]
            num_2 = (temp[1,2]-temp[2,2])*grad[i,j]+temp[1,2]
            if not (img_tidu[i,j]>num_1 and img_tidu[i,j]>num_2):
                flag = False
        #如果它此时为极大值，则保留
        if flag:
            img_yizhi[i,j] = img_tidu[i,j]
plt.figure(3)
plt.imshow(img_yizhi.astype(np.uint8),cmap='gray')
plt.title("非极大值抑制")

#########################4.双阈值检测######################
#设置低阈值和高阈值
lower_boundary = np.mean(img_yizhi)/2
high_boundary = lower_boundary*3
#设置栈,存储强边缘点
zhan = []
for i in range(1,img_yizhi.shape[0]-1):#外圈不考虑
    for j in range(1,img_yizhi.shape[1]-1):
        if img_yizhi[i,j] >= high_boundary:
            img_yizhi[i,j] = 255
            zhan.append([i,j])
        elif img_yizhi[i,j] <= lower_boundary:
            img_yizhi[i,j] = 0

#如果栈不为空，则处理弱边缘点
while len(zhan) != 0:
    #出栈
    temp_1, temp_2 = zhan.pop()
    temp = img_yizhi[temp_1-1:temp_1+2,temp_2-1:temp_2+2]
    #如果强边缘周围有高于低阈值的点，可将该点记为强边缘点
    for i in range(3):
        for j in range(3):
            if i==1 and j ==1:
                continue
            if (temp[i,j]<high_boundary) and (temp[i,j]>lower_boundary):
                img_yizhi[temp_1-1+i,temp_2-1+j] = 255
                zhan.append([temp_1-1+i,temp_2-1+j])
            

for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i,j] !=0 and img_yizhi[i,j]!=255:
            img_yizhi[i,j] = 0

plt.figure(4)
plt.imshow(img_yizhi.astype(np.uint8),cmap='gray')
plt.title('双阈值检测')

plt.show()


