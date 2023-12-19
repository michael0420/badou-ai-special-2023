import cv2
import numpy as np

if __name__ == '__main__':
    resized_image = cv2.imread("img1.jpg")
    resized_image2 = cv2.imread("img2.jpg")
    cv2.imshow("img1",resized_image),
    cv2.imshow("img2",resized_image2)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(resized_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(resized_image2, None)
    #关键点匹配
    bf = cv2.BFMatcher_create()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # 获取最佳匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
   # 获取匹配点的坐标
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵：
    homography_matrix, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

    #进行拼接
    result = cv2.warpPerspective(resized_image, homography_matrix, (resized_image.shape[1] + resized_image2.shape[1], resized_image.shape[0]))
    result[0:resized_image2.shape[0], 0:resized_image2.shape[1]] = resized_image2

    cv2.imshow("final result",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


