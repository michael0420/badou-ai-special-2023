import cv2
import random

def function(src,means,sigma,percetage):
    noiseImg = src
    noiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        noiseImg[randX,randY] = noiseImg[randX,randY] + random.gauss(means,sigma)
        if noiseImg[randX,randY] < 0:
            noiseImg[randX,randY] = 0
        elif noiseImg[randX,randY] > 255:
            noiseImg[randX, randY] = 255
    return noiseImg



if __name__ == '__main__':
    img = cv2.imread("lenna.png",0)
    img1 = function(img,2,4,0.8)
    img = cv2.imread('lenna.png')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("source",img2)
    cv2.imshow(" gaussian",img1)
    cv2.waitKey(0)
