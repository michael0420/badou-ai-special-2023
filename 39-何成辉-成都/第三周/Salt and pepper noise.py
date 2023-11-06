import cv2
import random


# 定义一个名为fun1的函数，接收两个参数：src（源图像）和percetage（噪声比例）。
def fun1(src, percetage):
		# 将源图像赋值给NoiseImg。
		NoiseImg = src
		# 计算需要添加的噪声像素数量，即src图像的像素数量乘以噪声比例。
		NoiseNum = int(percetage*src.shape[0]*src.shape[1])
		# 使用for循环，循环次数为噪声像素数量。
		for i in range(NoiseNum):
			# 每次取一个随机点,随机生成一个行坐标randX和一个列坐标randY。
			# 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
			# random.randint生成随机整数
			# 椒盐噪声图片边缘不处理，故-1
			randX = random.randint(0,src.shape[0]-1)
			randY = random.randint(0,src.shape[1]-1)
			# random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
			# 随机生成一个浮点数，如果小于等于0.5，则将NoiseImg对应位置的像素值设为0（黑色），否则设为255（白色）。
			if random.random() <= 0.5:
				NoiseImg[randX, randY] = 0
			else:
				NoiseImg[randX,randY] = 255
		# 返回处理后的图像NoiseImg。
		return NoiseImg


img = cv2.imread('lenna.png', 0)
# 调用fun1函数，传入img和噪声比例0.2，得到加噪后的图像img1。
img1 = fun1(img, 0.2)
# 在文件夹中写入命名为lenna_PepperandSalt.png的加噪后的图片
# cv2.imwrite('lenna_PepperandSalt.png',img1)

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('lenna_PepperandSalt',img1)
cv2.waitKey(0)

