import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = [u'SimHei']

def gray_Binarization(img):
    H, W, C = img.shape

    img_grey= np.zeros([H, W], img.dtype)
    img_erzhi = np.zeros([H, W], img.dtype)
    # cv2.imread用以下转换
    # img_grey = img_grey.astype(np.uint8)

    for i in range(H):
        for j in range(W):
            img_grey[i, j] = img[i, j, 0] * 0.11 + img[i, j, 1] * 0.59 + img[i, j, 2] * 0.3
            if img_grey[i, j] > 0.5:
                img_erzhi[i, j] = 1
            else:
                img_erzhi[i, j] = 0
    return img_grey, img_erzhi

if __name__ == '__main__':
    img_input = plt.imread('lenna.png')
    img_grey, img_erzhi = gray_Binarization(img_input)

    plt.subplots_adjust(hspace=0.5)
    plt.subplot(221), plt.title("原图")
    plt.imshow(img_input)

    plt.subplot(222), plt.title("灰度图")
    plt.imshow(img_grey, cmap='gray')

    plt.subplot(223), plt.title("二值图")
    plt.imshow(img_erzhi, cmap='gray')
    plt.show()
