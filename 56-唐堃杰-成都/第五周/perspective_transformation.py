# 透视变换
import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('./photo1.jpg')
    # 复制图片数据
    result3 = img.copy()
    # 原图的四个顶点
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    # 期望得到的4个顶点
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    print(img.shape)
    # 生成透视变换矩阵
    m = cv2.getPerspectiveTransform(src, dst)
    print("warpMatrix:")
    print(m)
    # 进行透视变换
    result = cv2.warpPerspective(result3, m, (337, 488))
    # 打印原图与透视变换后的图片
    cv2.imshow("src", img)
    cv2.imshow("result", result)
    cv2.waitKey(0)