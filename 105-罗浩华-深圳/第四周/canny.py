import cv2

img=cv2.imread("lenna.png")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray_img",gray_img)
cv2.waitKey()
cv2.imshow("canny",cv2.Canny(gray_img,200,100))
cv2.waitKey()
cv2.destroyAllWindows()