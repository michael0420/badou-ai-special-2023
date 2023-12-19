import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = [u'SimHei']

def Proximity_interpolation(img, H_out, W_out):
    H, W, C = img.shape
    img_output = np.zeros((H_out, W_out, C), img.dtype)
    H_ratio, W_ratio = H_out/H, W_out/W

    for i in range(H_out):
        for j in range(W_out):
            x, y = int(i/H_ratio + 0.5), int(j/W_ratio + 0.5)
            img_output[i, j] = img[x, y]

    return img_output

if __name__ == '__main__':
    img_input = plt.imread('lenna.png')
    img_output = Proximity_interpolation(img_input, 1000, 1000)

    plt.subplot(121), plt.title("原图(512x512)")
    plt.imshow(img_input,)

    plt.subplot(122), plt.title("临近插值图(1000x1000)")
    plt.imshow(img_output, cmap='gray')
    plt.show()
