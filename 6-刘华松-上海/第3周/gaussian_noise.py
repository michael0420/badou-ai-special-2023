
#实现高斯噪声***************************


import cv2

# 读取图像
image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

# 添加高斯噪声
mean = 0
var = 100
noisy_image = add_gaussian_noise(image, mean, var)

# 显示原始图像和添加高斯噪声后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)


def add_gaussian_noise(image, mean, var):
    row, col = image.shape
    sigma = var ** 0.5
    gaussian_noise = np.random.normal(mean, sigma, (row, col))
    noisy_image = image + gaussian_noise
    # 将像素值限制在 [0, 255] 区间内
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image
	
	
	
#其它代码****************

#gaussian_noise = np.random.normal(mean, sigma, (512, 512))
#noisy_image = cv2.add(image, gaussian_noise)	