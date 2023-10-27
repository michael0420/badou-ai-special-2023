import cv2
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def grayscale(image):
    height, width = image.shape[:2]
    img_gray = np.zeros((height, width), dtype=np.uint8)  # Declare img_gray as a numpy array
    for i in range(height):
        for j in range(width):
            m = image[i, j]
            img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # BGR
    return img_gray

def binarize(img_gray, threshold):
    image_binary = np.where(img_gray >= threshold, 255, 0).astype(np.uint8)
    return image_binary

# Read the image
image = cv2.imread('lenna.png')

# Convert to grayscale
img_gray = grayscale(image)

# Binarize
threshold = 128
image_binary = binarize(img_gray, threshold)

# Display the images
cv2.imshow('Original Image', image)
cv2.imshow('img_gray', img_gray)
cv2.imshow('Binarized Image', image_binary)

cv2.waitKey(0)
cv2.destroyAllWindows()
