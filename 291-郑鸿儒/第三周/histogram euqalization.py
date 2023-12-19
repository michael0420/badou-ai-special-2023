#!/usr/bin/env python
# encoding: utf-8
import cv2
import matplotlib.pyplot as plt
import numpy as np


# img = cv2.imread("img/lenna.png", 0)
# hist_img = cv2.equalizeHist(img)
# hist = cv2.calcHist([hist_img], [0], None, [256], [0, 256])
#
# plt.figure()
# plt.hist(hist, color="red")
#
# cv2.imshow("compare", np.hstack((img, hist_img)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 中文
img_origin = cv2.imread("img/lenna.png")
chans = cv2.split(img_origin)
colors = ("blue", "green", "red")
k = 0
plt.figure()
new_img = None
for chan, color in zip(chans, colors):
    chan_hist_img = cv2.equalizeHist(chan)
    chan_hist = cv2.calcHist(chan_hist_img, [0], None, [256], [0, 256])
    k += 1
    plt.subplot(2, 2, k)
    plt.hist(chan_hist, color=color)
    if new_img is not None:
        new_img = cv2.merge((new_img, chan_hist_img))
    else:
        new_img = chan_hist_img
plt.show()
cv2.imshow("compare", np.hstack((img_origin, new_img)))
cv2.waitKey(0)
cv2.destroyAllWindows()
