import cv2
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 100, 200)

fig = plt.figure()
plt.title("canny "), plt.axis('off')
plt.imshow(canny, cmap='gray', vmin=0, vmax=255)  # 原始图像
plt.show()