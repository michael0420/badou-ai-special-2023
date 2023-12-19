import cv2
img = cv2.imread("lenna.png")
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d_SIFT.create()
keypoints, descritor = sift.detectAndCompute(img_g, None)
sift_keypoints = cv2.drawKeypoints(image=img,
                                   outImage=img,
                                   keypoints=keypoints,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 0, 0))
cv2.imshow("sift_keypoints", sift_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
