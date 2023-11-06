import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist��ֱ��ͼ���⻯
����ԭ�ͣ� equalizeHist(src, dst=None)
src��ͼ�����(��ͨ��ͼ��)
dst��Ĭ�ϼ���
'''

def grayscale(image):
    height, width = image.shape[:2]
    img_gray = np.zeros((height, width), dtype=np.uint8)  # Declare img_gray as a numpy array
    for i in range(height):
        for j in range(width):
            m = image[i, j]
            img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # BGR
    return img_gray


# ��ȡ�Ҷ�ͼ��
img = cv2.imread("lenna.png", 1)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = grayscale(img)
# cv2.imshow("image_gray", gray)

# �Ҷ�ͼ��ֱ��ͼ���⻯
dst = cv2.equalizeHist(gray)


# # ��ɫͼ��ֱ��ͼ���⻯
# img = cv2.imread("lenna.png", 1)
# # cv2.imshow("src", img)
#
# # ��ɫͼ����⻯,��Ҫ�ֽ�ͨ�� ��ÿһ��ͨ�����⻯
# (b, g, r) = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# # �ϲ�ÿһ��ͨ��
# result = cv2.merge((bH, gH, rH))
# cv2.imshow("dst_rgb", result)
#
# cv2.waitKey(0)


# ֱ��ͼ
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

'''
images������ͼ�񣬿����ǵ�ͨ�����ͨ��ͼ����������Ϊuint8��float32��
channels��ָ��Ҫ����ֱ��ͼ��ͨ���б����ڻҶ�ͼ��ͨ��ֵΪ[0]�����ڲ�ɫͼ�񣬿���ָ��ͨ��ֵΪ[0]��[1]��[2]���ֱ��Ӧ��ɫ����ɫ�ͺ�ɫͨ����
mask����ѡ����������ָ������Ȥ�����������Ҫ����������ΪNone��
histSize��ָ��ֱ��ͼ�Ĵ�С�����Ҷȼ����������
ranges��ָ������ֵ�ķ�Χ��ͨ������Ϊ[0, 256]��
'''

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))  #ƴ��
cv2.waitKey(0)
