import cv2
import numpy as np
# 透视变换

# 寻找顶点
def find_document_vertices(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray,(5, 5),0)
    # 膨胀操作
    dilate = cv2.dilate(blurred,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    # 边缘检测
    edged = cv2.Canny(dilate, 30, 120, 3)
    # 轮廓检测
    cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]if len(cnts) == 2 else cnts[1]
    doc_cnt=None
    if len(cnts) > 0:
        # 根据轮廓面积从大到小排序
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            if len(approx) == 4:
                # doc_cnt = approx
                doc_cnt = approx.astype(np.float32)
                break
    return doc_cnt
if __name__=='__main__':
    img_path = 'photo1.jpg'
    document_vertices = find_document_vertices(img_path)
    image = cv2.imread('photo1.jpg')
    img_copy = image.copy()
    if document_vertices is not None and len(document_vertices) == 4:
        print(document_vertices)
        src_pts = np.float32([document_vertices[0],document_vertices[3],document_vertices[1],document_vertices[2]])
        # src_pts = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
        # dst_pts = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
        # dst_pts = np.float32([[0, 0], [337, 0], [337, 488], [0, 488]])
        # 获取源图像的宽度和高度
        src_height, src_width = img_copy.shape[:2]
        # 设置透视变换的目标图像尺寸
        dst_width = 337
        dst_height = int(src_height / src_width * dst_width)
        dst_pts = np.float32([[0, 0], [dst_width, 0], [0, dst_height], [dst_width, dst_height]])
        #
        # 生成透视变换矩阵
        pt = cv2.getPerspectiveTransform(src_pts, dst_pts)
        print('透视变换后的矩阵')
        print(pt)
        # 应用透视变换
        res = cv2.warpPerspective(img_copy, pt, (337, 488))
        cv2.imshow('src',image)
        cv2.imshow('PerspectiveTransform', res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('未找到或找到的顶点不足4个，无法进行透视变换。')
