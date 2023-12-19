import cv2
import numpy as np
import random
def SaltPepperNoise(src, percentage):
    NoiseImg = np.copy(src)
    NoiseNum = int(percentage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)
        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg
img = cv2.imread('lenna.png',0)
img1 = SaltPepperNoise(img, 0.8)
cv2.imshow('Stalt', np.hstack((img, img1)))
cv2.waitKey(0)
cv2.destroyAllWindows()