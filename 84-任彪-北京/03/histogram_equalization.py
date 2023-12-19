import  cv2
import numpy as np
import matplotlib.pyplot as plt
import collections

"""
以下是通过自己实现的获取均衡直方图
以下方法是调用api获取均衡结果
dst = cv2.equalizeHist(gray)
"""


def ownimplement(img,isRgb):
    #先获取宽高
    H, W = img.shape[:2]
    if(isRgb):
        #拆分多个通道的值
        cv2.imshow("source", img)

        # CV2读入的通道为bgr
        (b,g,r) = cv2.split(img)
        # 分别计算每个通道的均衡值
        #对应 new_b = cv2.equalizeHist(b)
        new_b = jisuan(b, H, W)
        new_g = jisuan(g, H, W)
        new_r = jisuan(r, H, W)
        result = cv2.merge((new_b, new_g, new_r))
        cv2.imshow("均衡化的结果", result)
        cv2.waitKey(0)
    else:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow("source", gray)
        new_gray = jisuan(gray, H, W )
        cv2.imshow("均衡化的结果", new_gray)
        cv2.waitKey(0)
"""
自己手写直方图均衡化的算法
步骤：
1、获取所有像素点的个数，
2、按照从小到大一次排序
3、对每个像素点，利用均衡直方图公式计算得到新的像素值。  q = sun(k) * 256 /( h *w )  -1 
为什么-1？ 因为从0开始计算的，q代表的是index的值
"""
def jisuan(channel, h, w):
    one_data = channel.ravel()
    dict = collections.Counter(one_data)
    keys = sorted(dict)
    new_dict = {}
    num_count = 0
    for i in keys:
        num_count = dict.get(i) + num_count;
        new_value = int(num_count*256/(h*w) - 1)
        # 防止出现超出 【0，255】的情况
        if(new_value < 0):
            new_value = 0
        if(new_value > 255):
            new_value = 255
        new_dict.setdefault(i, new_value)
    # 针对多为数组,获取数组的下标和值
    for index, x  in np.ndenumerate(channel):
        channel[index] = new_dict.get(x)
    return channel



"""
以下是通过api调用获取均衡直方图
"""
def apiGetHistogram(img,isGray):
    if(isGray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = cv2.equalizeHist(gray)
        # 多维度转一维度
        date = dst.ravel()
        # #使用cv2，和plot 一起使用
        # hist = cv2.calcHist([date], [0], None, [256], [0, 256])
        plt.figure()
        plt.hist(date, 256)
       # plt.plot(hist)
        plt.show()
    else:
        cv2.imshow("Original", img)
        # cv2.waitKey(0)

        chans = cv2.split(img)
        colors = ("b", "g", "r")
        plt.figure()
        plt.title("Flattened Color Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")

        for (chan, color) in zip(chans, colors):
            dst = cv2.equalizeHist(chan)
            hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
        plt.show()


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    # 传入 True 为 rgb 三通道均衡转化 ， 传入False 只均衡转化灰度直方图
    ownimplement(img,True)
    #apiGetHistogram(img,False)




