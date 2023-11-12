import cv2
import random

"""
椒盐噪声

分别实现了灰度图和彩色图三通道的加椒盐噪声
"""

def spiced_salt(img, scale, isrgb):
    #1、获取需要加椒盐的像素点个数
    salt_num = int(img.shape[0] * img.shape[1] * scale)
    result_img = None
    if(isrgb):
        result_img = img
        chans = cv2.split(img)
        new_chans = []
        for channel in chans:
            for i in range(salt_num):
                result = random.sample(range(0,img.shape[0]),2)
                if random.random() <= 0.5:
                    channel[result[0],result[1]] = 0
                else:
                    channel[result[0], result[1]] = 255
                new_chans.append(channel)
        # 三通道进行合并成新的图，不能使用原理的channel，不知道原理？
        #img = cv2.merge((new_chans[0],new_chans[1],new_chans[2]))
        result_img = cv2.merge((new_chans[0], new_chans[1], new_chans[2]))
    else:
        result_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray',result_img)
        for i in range(salt_num):
            #取不同的两个值，加大难度
            result = random.sample(range(0, result_img.shape[0]), 2)
            if random.random() <= 0.5:
                result_img[result[0], result[1]] = 0
            else:
                result_img[result[0], result[1]] = 255

    return result_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    # 传入 True 为 rgb 三通道加椒盐 ， 传入False 只均灰度图加椒盐
    new_img = spiced_salt(img,0.1,True)
    cv2.imshow('source',img)
    cv2.imshow('lenna_Salt',new_img)
    cv2.waitKey(0)