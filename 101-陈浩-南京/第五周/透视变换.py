import  cv2
import numpy as np

#加载图片
img = cv2.imread('photo1.jpg')

result3 = img.copy() #复制图片

#src为原图的四个顶点坐标
#dst为目标图的四个顶点坐标
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)

#生成透视变化矩阵，进行透视变换
m = cv2.getPerspectiveTransform(src,dst)
#打印透视变换矩阵
print("wrapMatrix:")
print(m)
#第一个参数图片，第二参数：变换矩阵，第三个参数：图片尺寸
result = cv2.warpPerspective(result3,m,(337,488))
cv2.imshow('src',img)
cv2.imshow('result',result)
cv2.waitKey(0)
