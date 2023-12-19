# 作业1：实现透视变换
# 1、已知变换前点(x0,y0),(x1,y1),(x2,y2),(x3,y3)
#     和变换后点(x0',y0'),(x1',y1'),(x2',y2'),(x3',y3')
# 2、通过 opencv的getPerspectiveTransform(src, dst) 获得 wrapMatrix
# 3、通过 opencv的 warpPerspective 使用 wrapMatrix 完成透视变换得到结果result

import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('photo1.jpg')
    src = [[207, 151], [517, 285], [17, 601], [343, 731]]
    src = np.float32(src)
    # 指定数组元素的数据类型为32位浮点数

    dst = [[100, 100], [437, 100], [100, 588], [437, 588]]
    dst = np.float32(dst)

    # 传入两个32位浮点数
    warpMatrix = cv2.getPerspectiveTransform(src, dst)
    print(warpMatrix)

    # 超过图像原来的长宽则用黑色补充，看上去像是图像的旋转
    result = cv2.warpPerspective(img, warpMatrix, (800, 1000))
    cv2.imshow("src", img)
    cv2.imshow("result", result)
    cv2.waitKey(0)