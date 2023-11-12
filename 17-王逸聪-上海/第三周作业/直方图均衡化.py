import cv2
import numpy as np
from matplotlib import pyplot as plt
if __name__ == '__main__':
    # 灰度图像直方图
    # img = cv2.imread("lenna.png")
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # dst = cv2.equalizeHist(gray)
    # cv2.imshow("compare",np.hstack([gray,dst]))
    # hist = cv2.calcHist([dst],[0],None,[256],[0,256])
    # plt.figure()
    # plt.hist(hist.ravel(),256)
    # plt.show()


    # 彩色直方图
    img = cv2.imread("lenna.png")

    (b,g,r) = cv2.split(img)
    bh = cv2.equalizeHist(b)
    gh = cv2.equalizeHist(g)
    rh = cv2.equalizeHist(r)
    dst = cv2.merge((bh,gh,rh))
    cv2.imshow("rgb_dst",np.hstack([img,dst]))
    cv2.waitKey(0)