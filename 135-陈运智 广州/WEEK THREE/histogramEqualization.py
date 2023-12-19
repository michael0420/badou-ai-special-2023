import cv2
import matplotlib.pyplot as plt
import numpy as np

img =cv2.imread("lenna.png")
h,w =img.shape[:2]
img_gary =np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gary[i,j]=int(m[0]*0.11 +m[1]*0.59 +m[2]*0.3)

# equalized_image = cv2.equalizeHist(img_gary)
# hist = cv2.calcHist([equalized_image],[0],None,[256],[0,256])
# plt.Figure()
# # plt.hist(img_gary.ravel(),256)
# plt.hist(equalized_image.ravel(),256)
# plt.show()
# cv2.imwrite("equalized.png",equalized_image)

hist = cv2.calcHist([img_gary],[0],None,[256],[0,256])
cdf = hist.cumsum()
cdf_min =cdf.min()
cdf_max =cdf.max()
cdf_mapping = ((cdf-cdf_min)/(h*w-1))*255
equalized_image = np.interp(img_gary, np.arange(256), cdf_mapping).astype(np.uint8)
plt.Figure()
# plt.hist(img_gary.ravel(),256)
plt.hist(equalized_image.ravel(),256)
plt.show()
cv2.imwrite("equalized.png",equalized_image)