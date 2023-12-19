import cv2

def detect_sift_keypoints(image_path):
    # 读取图像
    img = cv2.imread(image_path)

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建SIFT检测器对象
    sift = cv2.SIFT_create()

    # 检测关键点和计算特征描述符
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # 绘制关键点
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)

    # 显示图像
    cv2.imshow("Image with SIFT Keypoints", img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 替换为你的图像文件路径
image_path = "lenna.png"

# 调用函数进行SIFT特征点检测
detect_sift_keypoints(image_path)
