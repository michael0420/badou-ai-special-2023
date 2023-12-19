import cv2
from skimage import util

img = cv2.imread("img/lenna.png")
noise_img = util.random_noise(img, mode="localvar", clip=True)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# noise_img = util.random_noise(gray, mode="pepper")
cv2.imshow("src", img)
cv2.imshow("noise", noise_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
