import cv2
import numpy as np

#检测关键点
#读入图片

img=cv2.imread("lenna.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#调接口
sift=cv2.xfeatures2d.SIFT_create()

keypoints,descriptor=sift.detectAndCompute(gray,None)
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))

cv2.imshow("sift_keypoint",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

