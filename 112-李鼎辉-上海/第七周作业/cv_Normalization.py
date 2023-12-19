import cv2
import numpy as np
from matplotlib import pyplot as plt

def z_score_normalization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalization_data = (data - mean)/std
    return normalization_data
def batch_normalization(data, epsilon=1e-8):
    mean = np.mean(data, axis=0)
    variance = np.var(data, axis=0)
    normalization_data = (data-mean)/np.sqrt(variance+epsilon)
    return normalization_data
def min_max_normalization(data, min_val=0, max_val=1):
    min_data = np.min(data, axis=0)
    max_data = np.max(data, axis=0)
    normalization_data = min_val+(data - min_data)*(max_val - min_val) / (max_data - min_data)
    return normalization_data
if __name__ =="__main__":
    img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
    n_z_score = z_score_normalization(img)
    print(n_z_score)
    print('\n')
    n_batch = batch_normalization(img)
    print(n_batch)
    print('\n')
    n_min_max = min_max_normalization(img)
    print(n_min_max)
    print('\n')

    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(n_z_score, cmap='gray')
    plt.title('Z-Score Normalization')

    plt.subplot(132)
    plt.imshow(n_batch, cmap='gray')
    plt.title('Batch Normalization')

    plt.subplot(133)
    plt.imshow(n_min_max, cmap='gray')
    plt.title('Min-Max Normalization')

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()