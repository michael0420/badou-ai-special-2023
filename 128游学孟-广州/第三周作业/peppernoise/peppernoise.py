import numpy as np
import cv2

# 加载图像
image = cv2.imread('D://xuexi//zuoye//week3//peppernoise//pappernoise.webp')

def addpeppernoise(image,density):
    height,width,channels = image.shape
    noise_image = np.copy(image)

    #计算要添加噪声的像素数
    num_noisepixels = int(height*width*density)

    #在随机位置添加白色和黑色噪声
    for _ in range(num_noisepixels // 2):
        x = np.random.randint(0,width)
        y = np.random.randint(0,height)

        #添加白色噪声
        noise_image[y,x,:] = (255,255,255)

    for _ in range(num_noisepixels // 2):
        x = np.random.randint(0,width)
        y = np.random.randint(0,height)

        #添加黑色噪声
        noise_image[y,x,:] = (0,0,0)
    return noise_image

density = 0.02 #设置噪声密度

noisy_image = addpeppernoise(image,density)
cv2.imshow("peppernoise",image)
cv2.imshow("peppernoisenoisy",noisy_image)
cv2.imwrite('peppernoisenoisy.jpg',noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()