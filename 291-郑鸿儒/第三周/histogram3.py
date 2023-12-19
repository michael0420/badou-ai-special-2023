import cv2
import matplotlib.pyplot as plt


img = cv2.imread("img/lenna.png", 1)
chans = cv2.split(img)
colors = ("blue", "green", "red")
plt.figure()
k = 0
for chan, color in zip(chans, colors):
    print(color)
    # plt.hist(chan.ravel(), 256, color=color)
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    k += 1
    print(k)
    plt.subplot(2, 2, k)
    plt.plot(hist, color=color)
plt.show()
