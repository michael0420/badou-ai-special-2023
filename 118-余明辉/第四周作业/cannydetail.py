import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

#1,灰度化
img = cv2.imread('lenna.png')
im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#2，高斯平滑
#高斯核函数 g(x,y) = 1/(2*pi*sigma^2)*exp(-1*(x^2+y^2)/2/sigma^2)
sigma = 0.6
dim = 5
gauss_filter = np.zeros([dim,dim])
c1 = 1/(2*math.pi*sigma**2)
c2 = -1/(2*sigma**2)
x = [i-dim//2 for i in range(dim)] #初始序列
for i in range(dim):
    for j in range(dim):
        gauss_filter[i,j] = c1*math.exp(c2*(x[i]**2+x[j]**2))
gauss_filter = gauss_filter/gauss_filter.sum()

#使用高斯核函数
dx,dy = im.shape
im_new = np.zeros([dx,dy])
#进行边界填充然后卷积
tmp = dim//2
im_pad = np.pad(im, ((tmp, tmp), (tmp, tmp)), 'constant')
for i in range(dx):
    for j in range(dy):
        im_new[i,j] = np.sum(im_pad[i:i+dim,j:j+dim]*gauss_filter)
plt.figure(1)
plt.imshow(im_new.astype(np.uint8), cmap='gray')

#3,sobel,x和y两个方向使用sobel算子
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
im_x = np.zeros([dx,dy])
im_y = np.zeros([dx,dy])
imm = np.zeros([dx,dy])
imm_pad = np.pad(im_new, ((1, 1),(1, 1)), 'constant')
for i in range(dx):
    for j in range(dy):
        im_x[i,j] = np.sum(imm_pad[i:i+3, j:j+3]*sobel_kernel_x)
        im_y[i, j] = np.sum(imm_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
        imm[i,j] = np.sqrt(im_x[i,j]**2+im_y[i,j]**2)
plt.figure(2)
plt.imshow(imm.astype(np.uint8), cmap='gray')
plt.axis('off')

#4，非极大值抑制
#每个点的梯度
im_x[im_x == 0] = 0.00000001
angle = im_y/im_x
im_yizhi = np.zeros([dx,dy])
for i in range(1,dx-1):
    for j in range(1,dy-1):
        f = 1
        t = imm[i-1:i+2,j-1:j+2]
        if angle[i,j]>=1:
            n1 = (t[0,2]-t[0,1])/angle[i,j] + t[0,1]
            n2 = (t[2,0] - t[2,1])/angle[i,j] + t[2,1]
            if not(imm[i,j] > n1 and imm[i,j] > n2):
                f = 0
        elif angle[i,j]<=-1:
            n1 = (t[0,1]-t[0,0])/angle[i,j] + t[0,1]
            n2 = (t[2,1] - t[2,2])/angle[i,j] + t[2,1]
            if not(imm[i,j] > n1 and imm[i,j] > n2):
                f = 0
        elif angle[i,j]>0:
            n1 = (t[0,2]-t[1,2])/angle[i,j] + t[1,2]
            n2 = (t[2,0] - t[1,0])/angle[i,j] + t[1,0]
            if not(imm[i,j] > n1 and imm[i,j] > n2):
                f = 0
        elif angle[i,j]<0:
            n1 = (t[1,0]-t[0,0])/angle[i,j] + t[1,0]
            n2 = (t[1,2] - t[2,2])/angle[i,j] + t[1,2]
            if not(imm[i,j] > n1 and imm[i,j] > n2):
                f = 0
        if f:
            im_yizhi[i,j] = imm[i,j]
plt.figure(3)
plt.imshow(im_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')

#5，双阈值检测
l = im_yizhi.mean() * 0.5
h = l * 3
zhan = []
for i in range(1, dx - 1):  # 外圈不考虑了
    for j in range(1, dy - 1):
        if im_yizhi[i, j] >= h:  # 取，一定是边的点
            im_yizhi[i, j] = 255
            zhan.append([i, j])
        elif im_yizhi[i, j] <= l:  # 舍
            im_yizhi[i, j] = 0

while not len(zhan) == 0:
    temp_1, temp_2 = zhan.pop()  # 出栈
    a = im_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
    kk = [[0,1],[1,1],[1,0],[1,-1],[-1,0],[-1,-1],[0,-1],[1,-1]]
    for i in range(8):
        xx = temp_1 + kk[i][0]
        yy = temp_2 + kk[i][1]
        if(im_yizhi[xx,yy]>l and im_yizhi[xx,yy]<h):
            im_yizhi[xx,yy] = 255
            zhan.append([xx,yy])

for i in range(dx):
    for j in range(dy):
        if im_yizhi[i, j] != 0 and im_yizhi[i, j] != 255:
            im_yizhi[i, j] = 0

plt.figure(4)
plt.imshow(im_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')
plt.show()
