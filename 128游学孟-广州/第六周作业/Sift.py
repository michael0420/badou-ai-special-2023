import cv2

# 读取源图像和目标图像
img1 = cv2.imread("D://xuexi//zuoye//week6//img1.jpg",cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("D://xuexi//zuoye//week6//img2.jpg",cv2.IMREAD_GRAYSCALE)

# 缩放图像
scale_percent = 50
width1 = int(img1.shape[1]*scale_percent/100)
height1 = int(img1.shape[0]*scale_percent/100)
dim1 = (width1,height1)
img1 = cv2.resize(img1,dim1,interpolation=cv2.INTER_AREA)

width2 = int(img2.shape[1]*scale_percent/100)
height2 = int(img2.shape[0]*scale_percent/100)
dim2 = (width2,height2)
img2 = cv2.resize(img2,dim2,interpolation=cv2.INTER_AREA)

# 创建SIFT对象
sift = cv2.SIFT_create()

# 检测关键点和计算描述符
keypoints1,descriptors1 = sift.detectAndCompute(img1,None)
keypoints2,descriptors2 = sift.detectAndCompute(img2,None)

# 创建BFMatcher对象，并进行特征点匹配
bf = cv2.BFMatcher()
matches = bf.match(descriptors1,descriptors2)

# 根据特征点间的距离排序
matches = sorted(matches,key = lambda x:x.distance)

# 可视化匹配结果
img_matches = cv2.drawMatches(img1,keypoints1,img2,keypoints2,matches[:10],None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Matches",img_matches)
cv2.waitKey(0)