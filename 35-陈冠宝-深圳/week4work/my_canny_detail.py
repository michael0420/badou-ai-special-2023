import numpy as np
import matplotlib.pyplot as plt
import math
import cv2


class CCANNY(object):

    def __init__(self, img,gauss_sigma,gauss_size,low_thread,high_thread):
        self.src = img#第一步传入灰度图
        self.gauss_sigma=gauss_sigma
        self.gauss_size=gauss_size
        self.lower_boundary=low_thread
        self.high_boundary=high_thread

        dx, dy = self.src.shape
        self.dx = dx
        self.dy = dy

        self.gaussimg=self._gauss()#第二步高斯平滑
        self.img_tidu,self.angle=self._tidu()#第三步soble求梯度
        self.img_yizhi=self._yizhi()#第四步非极大抑制
        self.img_canny=self._canny()#第五步双阈值检测和边缘连接



    def _gauss(self):
        # sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
        # dim = int(np.round(6 * sigma + 1))  # round是四舍五入函数，根据标准差求高斯核是几乘几的，也就是维度
        if self.gauss_size % 2 == 0:  # 最好是奇数,不是的话加一
            self.gauss_size += 1
        Gaussian_filter = np.zeros([self.gauss_size, self.gauss_size])  # 存储高斯核，这是数组不是列表了
        tmp = [i - self.gauss_size // 2 for i in range(self.gauss_size)]  # 生成一个序列
        n1 = 1 / (2 * math.pi * self.gauss_sigma ** 2)  # 计算高斯核
        n2 = -1 / (2 * self.gauss_sigma ** 2)
        for i in range(self.gauss_size):
            for j in range(self.gauss_size):
                Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
        Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()

        mygaussimg = np.zeros(self.src.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
        tmp = self.gauss_size // 2
        img_pad = np.pad(self.src, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补
        for i in range(self.dx):
            for j in range(self.dy):
                mygaussimg[i, j] = np.sum(img_pad[i:i + self.gauss_size, j:j + self.gauss_size] * Gaussian_filter)

        return mygaussimg

    def _tidu(self):
        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        img_tidu_x = np.zeros(self.gaussimg.shape)  # 存储梯度图像
        img_tidu_y = np.zeros([self.dx, self.dy])
        img_tidu = np.zeros(self.gaussimg.shape)
        img_pad = np.pad(self.gaussimg, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
        for i in range(self.dx):
            for j in range(self.dy):
                img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)  # x方向
                img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)  # y方向
                img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)
        img_tidu_x[img_tidu_x == 0] = 0.00000001
        angle = img_tidu_y/img_tidu_x
        return img_tidu,angle

    def _yizhi(self):
        img_yizhi = np.zeros(self.img_tidu.shape)
        for i in range(1, self.dx-1):
            for j in range(1, self.dy-1):
                flag = True  # 在8邻域内是否要抹去做个标记
                temp = self.img_tidu[i-1:i+2, j-1:j+2]  # 梯度幅值的8邻域矩阵
                if self.angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                    num_1 = (temp[0, 1] - temp[0, 0]) / self.angle[i, j] + temp[0, 1]
                    num_2 = (temp[2, 1] - temp[2, 2]) / self.angle[i, j] + temp[2, 1]
                    if not (self.img_tidu[i, j] > num_1 and self.img_tidu[i, j] > num_2):
                        flag = False
                elif self.angle[i, j] >= 1:
                    num_1 = (temp[0, 2] - temp[0, 1]) / self.angle[i, j] + temp[0, 1]
                    num_2 = (temp[2, 0] - temp[2, 1]) / self.angle[i, j] + temp[2, 1]
                    if not (self.img_tidu[i, j] > num_1 and self.img_tidu[i, j] > num_2):
                        flag = False
                elif self.angle[i, j] > 0:
                    num_1 = (temp[0, 2] - temp[1, 2]) * self.angle[i, j] + temp[1, 2]
                    num_2 = (temp[2, 0] - temp[1, 0]) * self.angle[i, j] + temp[1, 0]
                    if not (self.img_tidu[i, j] > num_1 and self.img_tidu[i, j] > num_2):
                        flag = False
                elif self.angle[i, j] < 0:
                    num_1 = (temp[1, 0] - temp[0, 0]) * self.angle[i, j] + temp[1, 0]
                    num_2 = (temp[1, 2] - temp[2, 2]) * self.angle[i, j] + temp[1, 2]
                    if not (self.img_tidu[i, j] > num_1 and self.img_tidu[i, j] > num_2):
                        flag = False
                if flag:
                    img_yizhi[i, j] = self.img_tidu[i, j]
        return img_yizhi

    def _canny(self):
        cannyimg = cv2.copyTo(src=self.img_yizhi, mask=None)
        zhan = []
        for i in range(1, cannyimg.shape[0] - 1):  # 外圈不考虑了
            for j in range(1, cannyimg.shape[1] - 1):
                if cannyimg[i, j] >= self.high_boundary:  # 取，一定是边的点
                    cannyimg[i, j] = 255
                    zhan.append([i, j])
                elif cannyimg[i, j] <= self.lower_boundary:  # 舍
                    cannyimg[i, j] = 0

        while not len(zhan) == 0:
            temp_1, temp_2 = zhan.pop()  # 出栈
            a = cannyimg[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
            if (a[0, 0] < self.high_boundary) and (a[0, 0] > self.lower_boundary):
                cannyimg[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
                zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
            if (a[0, 1] < self.high_boundary) and (a[0, 1] > self.lower_boundary):
                cannyimg[temp_1 - 1, temp_2] = 255
                zhan.append([temp_1 - 1, temp_2])
            if (a[0, 2] < self.high_boundary) and (a[0, 2] > self.lower_boundary):
                cannyimg[temp_1 - 1, temp_2 + 1] = 255
                zhan.append([temp_1 - 1, temp_2 + 1])
            if (a[1, 0] < self.high_boundary) and (a[1, 0] > self.lower_boundary):
                cannyimg[temp_1, temp_2 - 1] = 255
                zhan.append([temp_1, temp_2 - 1])
            if (a[1, 2] < self.high_boundary) and (a[1, 2] > self.lower_boundary):
                cannyimg[temp_1, temp_2 + 1] = 255
                zhan.append([temp_1, temp_2 + 1])
            if (a[2, 0] < self.high_boundary) and (a[2, 0] > self.lower_boundary):
                cannyimg[temp_1 + 1, temp_2 - 1] = 255
                zhan.append([temp_1 + 1, temp_2 - 1])
            if (a[2, 1] < self.high_boundary) and (a[2, 1] > self.lower_boundary):
                cannyimg[temp_1 + 1, temp_2] = 255
                zhan.append([temp_1 + 1, temp_2])
            if (a[2, 2] < self.high_boundary) and (a[2, 2] > self.lower_boundary):
                cannyimg[temp_1 + 1, temp_2 + 1] = 255
                zhan.append([temp_1 + 1, temp_2 + 1])

        for i in range(cannyimg.shape[0]):
            for j in range(cannyimg.shape[1]):
                if cannyimg[i, j] != 0 and cannyimg[i, j] != 255:
                    cannyimg[i, j] = 0
        return cannyimg





if __name__ == '__main__':
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    # img = img.mean(axis=-1)  # 取均值就是灰度化了
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    cannyimg=CCANNY(img,0.5,5,22,66)

    plt.figure(1)
    # plt.imshow(img.astype(np.uint8), cmap='gray')
    plt.imshow(cannyimg.src.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.figure(2)
    # gaussimg=cannyimg.gaussimg.astype(np.uint8)
    plt.imshow(cannyimg.gaussimg.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.figure(3)
    plt.imshow(cannyimg.img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')

    plt.figure(4)
    plt.imshow(cannyimg.img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.figure(5)
    plt.imshow(cannyimg.img_canny.astype(np.uint8), cmap='gray')
    plt.axis('off')


    plt.show()