import cv2
import numpy as np
import FindVertex as fv


img = cv2.imread('photo1.jpg')

# 这行代码创建了img的一个副本，并将其存储在变量result3中。这样做的目的是为了避免对原始图像进行任何修改。
result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''

# 这里要注意一下，我们的src和dst的四个轮廓顶点一定要对应，
# 如果你执行下面的代码，你会发现它并不是我们期望的结果。
# src = np.float32(FV.docCnt)
# dst = np.float32([[0, 0], [337, 488], 0, 488],  [337, 0]])

src = np.float32(fv.docCnt)

# 这行代码定义了一个目标区域的顶点坐标数组，并将其存储在变量dst中。这个数组表示一个矩形区域，其左上角坐标为(0, 0)，右下角坐标为(337, 488)。
dst = np.float32([[0, 0], [0, 488], [337, 488], [337, 0]])

print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)

# 这行代码使用OpenCV库的warpPerspective函数将result3图像按照透视变换矩阵m进行变换，并将结果存储在变量result中。变换后的图像大小为(337, 488)。
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
