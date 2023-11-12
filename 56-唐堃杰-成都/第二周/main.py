import cv2
from PIL import Image
if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    print(img_gray)
    pass
