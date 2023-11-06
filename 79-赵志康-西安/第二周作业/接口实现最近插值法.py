import cv2
src_img = cv2.imread("123.png")
dst_img = cv2.resize(src_img, (256,256))
cv2.imshow("src_img", src_img)
cv2.imshow("dst_img", dst_img)
cv2.waitKey()