import cv2

# 读取图像
# image = cv2.imread('D://xuexi//zuoye//week4//canny//cannyphoto.jpg')

#读取图片
image = cv2.imread('D://xuexi//zuoye//week4//canny//cannyphoto.jpg',0)

#噪声抑制 - 高斯滤波
blurred  = cv2.GaussianBlur(image,(5,5),0)
threshold1 = 20
threshold2 = 50

# 运行Canny算法
edges = cv2.Canny(blurred,threshold1,threshold2)

cv2.imshow('canny',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()