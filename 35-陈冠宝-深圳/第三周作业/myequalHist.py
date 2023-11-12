import cv2
import numpy as np
from matplotlib import pyplot as plt

def mycalcHist(src):
    calcHist = np.zeros(shape=(256,1))
    reflection = np.zeros(shape=(256,1))
    H=src.shape[0]
    W=src.shape[1]
    count=H*W
    accumulation=0
    for i in range(H):
        for j in range(W):
            value=src[i][j]
            calcHist[value][0]+=1
    for i in range(256):
        value=calcHist[i]/count
        accumulation+=value
        if accumulation!=0:
            reflection[i]=int(accumulation*256-1)
    return(calcHist,reflection)

def myHist(src):
    (src_calcHist,reflection)=mycalcHist(src)
    H=src.shape[0]
    W=src.shape[1]
    equalizeHist = np.zeros(shape=(H,W),dtype=int)
    for i in range(H):
        for j in range(W):
            value=src[i][j]
            equalizeHist[i][j]=int(reflection[value])
    (calcHist,reflection)=mycalcHist(equalizeHist)
    equalizeHist=equalizeHist.astype(np.uint8)
    return (equalizeHist,calcHist,src_calcHist)

length = 256  # 指定列表的长度
element = 0  # 指定重复的元素
my_list = [element] * length

img = cv2.imread("lenna.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

(b, g, r) = cv2.split(img)#rgb通道分离
(my_equalizeHist_gray, my_equalizeHist_gray_calcHist, my_gray_calcHist) = myHist(gray)#rgb通道分离
(my_equalizeHist_b, my_equalizeHist_b_calcHist, my_b_calcHist) = myHist(b)#rgb通道分离
(my_equalizeHist_g, my_equalizeHist_g_calcHist, my_g_calcHist) = myHist(g)#rgb通道分离
(my_equalizeHist_r, my_equalizeHist_r_calcHist, my_r_calcHist) = myHist(r)#rgb通道分离


#使用彩色直方图进行的均衡化
bH = cv2.equalizeHist(b)#均衡化
bH_hist = cv2.calcHist([bH], [0], None, [256], [0,256])#直方图
b_hist = cv2.calcHist([b], [0], None, [256], [0,256])#原图直方图
gH = cv2.equalizeHist(g)
gH_hist = cv2.calcHist([gH], [0], None, [256], [0,256])
g_hist = cv2.calcHist([g], [0], None, [256], [0,256])
rH = cv2.equalizeHist(r)
rH_hist = cv2.calcHist([rH], [0], None, [256], [0,256])
r_hist = cv2.calcHist([r], [0], None, [256], [0,256])
cv2dstimg=cv2.merge((bH,gH,rH))
myequalizeHist=cv2.merge((my_equalizeHist_b,my_equalizeHist_g,my_equalizeHist_r))
cv2dstgray = cv2.equalizeHist(gray)#使用内置库的均衡化
#展示图像和直方图
plt.figure(num='直方图',figsize=(12,12))
plt.subplot(3,2,1)
plt.title('原图直方图',fontproperties='SimHei')
plt.plot(b_hist,color='blue')
plt.plot(g_hist,color='green')
plt.plot(r_hist,color='red')
plt.subplot(3,2,2)
plt.title('灰度直方图',fontproperties='SimHei')
plt.hist(gray.ravel(), 256)
plt.subplot(3,2,3)
plt.title('cv2均衡化灰度直方图',fontproperties='SimHei')
plt.hist(cv2dstgray.ravel(), 256)
plt.subplot(3,2,4)
plt.title('cv2均衡化彩图直方图',fontproperties='SimHei')
plt.plot(bH_hist,color='blue')
plt.plot(gH_hist,color='green')
plt.plot(rH_hist,color='red')
plt.subplot(3,2,5)
plt.title('my均衡化灰度直方图',fontproperties='SimHei')
plt.hist(my_equalizeHist_gray.ravel(), 256)
plt.subplot(3,2,6)
plt.title('my均衡化彩图直方图',fontproperties='SimHei')
plt.plot(my_equalizeHist_b_calcHist,color='blue')
plt.plot(my_equalizeHist_g_calcHist,color='green')
plt.plot(my_equalizeHist_r_calcHist,color='red')
plt.show()
cv2.imshow("graycompare", np.hstack([gray,cv2dstgray, my_equalizeHist_gray]))
cv2.imshow("rgbcompare", np.hstack([img, cv2dstimg,myequalizeHist]))
cv2.waitKey(0)
