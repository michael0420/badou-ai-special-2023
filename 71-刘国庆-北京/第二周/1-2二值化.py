import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

plt.subplot(331)
img = plt.imread("lenna.png")
print(img)
plt.imshow(img)

plt.subplot(332)
img_gray = rgb2gray(img)
print(img_gray)
plt.imshow(img_gray, cmap='gray')

plt.subplot(333)
img_binary = np.where(img_gray >= 0.5, 1, 0)
print(img_binary)
plt.imshow(img_binary, cmap='gray')

plt.show()
