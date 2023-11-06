import cv2
import random
"""
椒盐噪声

分别实现了灰度图和彩色图三通道的加高斯噪声
"""


def add_gauss(img, scale, sigma, mean, isrgb):
    # 1、获取需要加高斯噪声的像素点个数
    gauss_num = int(img.shape[0] * img.shape[1] * scale)
    result_img = None
    if(isrgb):
        result_img = img
        chans = cv2.split(img)
        new_chans = []
        for channel in chans:
            for i in range(gauss_num):
            #计算高斯随机数
                rand_num = random.gauss(mean,sigma)
                random_h = random.randint(0,img.shape[0]-1)
                random_w = random.randint(0,img.shape[1]-1)
                channel[random_h,random_w] = channel[random_h,random_w] + rand_num
                if(channel[random_h,random_w] < 0):
                    channel[random_h, random_w] = 0
                if(channel[random_h,random_w] > 255):
                    channel[random_h, random_w] = 255
            new_chans.append(channel)
        result_img = cv2.merge((new_chans[0],new_chans[1],new_chans[2]))
    else:
        result_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray 图",result_img)
        for i in range(gauss_num):
            # 计算高斯随机数
            rand_num = random.gauss(mean, sigma)
            random_h = random.randint(0, img.shape[0] - 1)
            random_w = random.randint(0, img.shape[1] - 1)
            result_img[random_h, random_w] = result_img[random_h, random_w] + rand_num
            if (result_img[random_h, random_w] < 0):
                result_img[random_h, random_w] = 0
            if (result_img[random_h, random_w] > 255):
                result_img[random_h, random_w] = 255
    return result_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    # 传入 True 为 rgb 三通道加椒盐 ， 传入False 只均灰度图加椒盐
    new_img = add_gauss(img, 0.4, 100, 2, True)
    cv2.imshow('source',img)
    cv2.imshow('lenna_gaosi',new_img)
    cv2.waitKey(0)