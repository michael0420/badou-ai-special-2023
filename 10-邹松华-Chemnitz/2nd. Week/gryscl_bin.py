# Homework Assignment / Hausaufgabe vom Songhua Zou

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2



def grayscale_1(img):

    img = cv2.imread(img)
    height, weigh = img.shape[:2]
    # Get only the first two values.
    img_gry = np.zeros([height, weigh], img.dtype)
    # If we call the func. img.dtype (abbr. data type) like here, then the new blank img will be assigned with
    # all the infos from ori img. But with func. np.uint8, we will assign it with type int8 data.
    for y in range(height):
        for x in range(weigh):
            tmp = img[y, x]
            # Attention, the return value of func. img[] will be in order B, G, R

            img_gry[y, x] = int(tmp[0]*0.11 + tmp[1]*0.59 + tmp[2]*0.3)
    print(img_gry)
    print("image show gray: %s" % img_gry)
    cv2.imshow("image show gray", img_gry)
    cv2.waitKey(0)


def grayscale_2(img):

    plt.subplot(131)
    img = plt.imread(img)
    plt.imshow(img)
    print("--ori img--")
    print(img)

    plt.subplot(132)
    img_gry = rgb2gray(img)
    plt.imshow(img_gry, cmap='gray')
    print("--gray img--")
    print(img_gry)

    return img_gry


def bnry(img_gry):
    img_bnry = np.where(img_gry >= 0.5, 1, 0)
    print("--binary img--")
    print(img_bnry)
    print(img_bnry.shape)

    plt.subplot(133)
    plt.imshow(img_bnry, cmap="gray")
    plt.show()


if __name__ == "__main__":

    img_path = 'lenna.png'

    grayscale_1(img_path)

    gry_img = grayscale_2(img_path)

    bnry(gry_img)




