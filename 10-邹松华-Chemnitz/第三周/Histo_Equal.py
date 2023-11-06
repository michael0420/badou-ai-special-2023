'''

@author Songhua Zou
Dies ist eine der Hausaufgaben von Songhua.

Please do not apply "Ctrl C + V" magic to my code without my permission.
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np


###### Histogram Equalization for a gray scale image ######
def histo_equal_gry(img_path):

    # Get a gray scale image
    img = cv2.imread(img_path, 0)
    '''
    Actually I think, In this context, setting the flag to 0 for grayscale is more efficient.
    Considering that our goal is to perform histogram equalization,
    which will be applied on a grayscale image, reading the image directly in grayscale
    saves processing time as it bypasses the need to convert from color to grayscale later.
    
    Other flag values for reference:
    cv2.IMREAD_COLOR or 1: Reads the image in color mode. This is the default value.
    cv2.IMREAD_GRAYSCALE or 0: Reads the image in grayscale mode.
    cv2.IMREAD_UNCHANGED or -1: Reads the image, including the alpha transparency channel (if available).
    '''

    # cv2.imshow('Gray Scale', img)
    # cv2.waitKey(0)

    # Histogram Equalization
    # Normalized CDF ~~ (Cumulative Frequency * Maximum Pixel Value) / Total Number of Pixels
    dst = cv2.equalizeHist(img)

    # Histogram
    hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

    plt.figure()
    plt.hist(dst.ravel(), bins=256)
    plt.show()

    cv2.imshow('Gray Scale Image Compare Before & After Histogram Equalization', np.hstack([img, dst]))
    cv2.waitKey(0)

def histo_equal_clr(img_path):

    # Get a color image
    img = cv2.imread(img_path)
    # If we don't manually set the flag, then the default flag value is "1"

    # cv2.imshow('Color', img)
    # cv2.waitKey(0)

    # Histogram Equalization
    # Split the channels
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)

    # Merge the channels
    mrg = cv2.merge([bH, gH, rH])
    # Notice here! BIF merge need a list as a parameter!

    cv2.imshow('Color Image Compare Before & After Histogram Equalization', np.hstack([img, mrg]))
    cv2.waitKey(0)

if __name__ == '__main__':

    img_path = 'lenna.png'
    histo_equal_gry(img_path)
    histo_equal_clr(img_path)



