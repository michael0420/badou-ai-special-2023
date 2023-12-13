import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

x, y = load_iris(return_X_y=True)
# print(x,y)
pca = PCA(n_components=0.93)
pca.fit(x)
newX = pca.fit_transform(x)
print(pca.explained_variance_ratio_)
print(pca.n_components_)
print(newX)
