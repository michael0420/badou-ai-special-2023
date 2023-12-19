import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage import io, color

# 读取Lenna图像，并将图像转换为灰度图像
img = cv2.imread("C:/Users/15082/Desktop/lenna.png")
greyimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 执行PCA，主成分数量选了30
pca = PCA(n_components=30)
greyimg_pca = pca.fit_transform(greyimg)    # 将原始的二维灰度图像数组转换为一个行向量，以便用于 PCA 的处理

# 将PCA结果映射回到原本的数据空间
greyimg_pca_restored = pca.inverse_transform(greyimg_pca)   # 将数据映射回原数据空间

# 绘制原始图像和PCA处理后的图像
plt.figure(figsize=(16, 8))  # 创建一个绘图对象

plt.subplot(1, 2, 1)
plt.imshow(greyimg, cmap='gray')
plt.title('Original Grey Image')

plt.subplot(1, 2, 2)
plt.imshow(greyimg_pca_restored, cmap='gray')
plt.title('PCA Processed Image')

plt.show()


