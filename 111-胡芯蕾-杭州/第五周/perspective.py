import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('photo1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(dilate, 30, 120, 3)

# Find contours
cnts= cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0]
docCnt = None

# Process contours
if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True) # 轮廓周长
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) # 近似多边形顶点
        if len(approx) == 4:
            docCnt = approx
            break
result = img.copy()
src = np.squeeze(docCnt).astype(np.float32)
dst = np.float32([[16, 200], [16, 603], [350, 603], [350, 200]])
m = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(result, m, (img.shape[1], img.shape[0]))

fig = plt.figure()
plt.subplot(121), plt.title("img"), plt.axis('off')
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(122), plt.title("Perspective"), plt.axis('off')
plt.imshow(result, cmap='gray', vmin=0, vmax=255)
plt.show()