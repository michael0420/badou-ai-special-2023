import cv2
import numpy as np

#读取图像
image = cv2.imread('D://xuexi//zuoye//week3//gaussnoise//guassnoise.webp')

#图像尺寸
height, width, channels = image.shape
#生辰高斯噪声
mean = 0 # 均值
staddev = 30 # 标准差

gaussian_noise = np.random.normal(mean,staddev,(height,width,channels))

# 添加噪声
noisy_image = image +gaussian_noise

# 限制图像像素值范围在0~255之间
noisy_image = np.clip(noisy_image,0,255).astype(np.uint8)

# 显示结果

cv2.imshow('gaussnoise',image)
cv2.imshow('gaussnoisenoisy',noisy_image)
#保存结果
cv2.imwrite('gaussnoisenoisy.jpg',noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()