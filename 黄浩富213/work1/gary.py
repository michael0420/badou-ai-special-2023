import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.color import rgb2gray

def cv2_get_gary(imge):
    img1 = cv2.imread(imge)
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    h = img2.shape[0]
    w = img2.shape[1]

    img_gary = np.zeros([h, w], img2.dtype)
    for i in range(h):
        for j in range(w):
            r = img2[i, j][0]
            g = img2[i, j][1]
            b = img2[i, j][2]
            img_gary[i, j] = int((r * 0.11) + (g * 0.59) + (b * 0.3))
    cv2.imshow("img_gary", img_gary)
    print("---img_gary lenna----")
    print(img_gary)
    cv2.waitKey(0)

def plt_get_gary_binary(imge):
    img = cv2.cvtColor(cv2.imread(imge), cv2.COLOR_BGR2RGB)
    plt.subplot(221)
    plt.imshow(img)
    print(img)
    print("---image source----")

    img_gray = rgb2gray(img)
    plt.subplot(222)
    plt.imshow(img_gray, cmap='gray')
    print(img_gray)
    print("---image gray----")

    h = img_gray.shape[0]
    w = img_gray.shape[1]
    print(h, w)
    img_binary = img_gray
    for i in range(h):
        for j in range(w):
            if img_gray[i, j] < 0.5:
                img_binary[i, j] = 0
            else:
                img_binary[i, j] = 1
    plt.subplot(223)
    plt.imshow(img_binary, cmap='gray')
    print(img_binary)
    print("---image binary----")
    plt.show()

if __name__ == '__main__':
    cv2_get_gary('lenna.png')
    plt_get_gary_binary('lenna.png')




