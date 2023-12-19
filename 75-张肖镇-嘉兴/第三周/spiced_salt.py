import random
import numpy as np
import cv2 as cv

ratio = 0.99


def salt(image):
    height, width = image.shape
    output = np.zeros(image.shape, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            rand = random.random()
            if rand > ratio:
                if random.random() > 0.5:
                    output[i, j] = 255
                else:
                    output[i, j] = 0
            else:
                output[i, j] = image[i, j]
    return output


if __name__ == "__main__":
    image = cv.imread("test.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    output = salt(image)
    cv.imshow("after", output)
    cv.waitKey()