### hist_color
彩色直方图对比

<img src="picture\image-20231102230131728.png" alt="image-20231102230131728" style="zoom:50%;" />

### hist_gray
灰度直方图对比

<img src="picture\image-20231102230208232.png" alt="image-20231102230208232" style="zoom:50%;" />


### gaussian_noise  
高斯噪声 思路是 取所有点的百分比的数量的点，随机进行像素点修改

<img src="picture\image-20231102230456976.png" alt="image-20231102230456976" style="zoom:50%;" />



### 椒盐噪声

椒盐噪声 思路是 取所有点的百分比的数量的点，随机进行像素点修改 （不是0就是255）

<img src="picture\image-20231102230332105.png" alt="image-20231102230332105" style="zoom: 50%;" />

### 均匀噪声

<img src="picture\image-20231102230731466.png" alt="image-20231102230731466" style="zoom:50%;" />

### 其他噪声使用方法

```
import cv2 as cv
import numpy as np
from PIL import Image
from skimage import util

'''
def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
功能：为浮点型图片添加各种随机噪声
参数：
image：输入图片（将会被转换成浮点型），ndarray型
mode： 可选择，str型，表示要添加的噪声类型
    gaussian：高斯噪声
    localvar：高斯分布的加性噪声，在“图像”的每个点处具有指定的局部方差。
    poisson：泊松噪声
    salt：盐噪声，随机将像素值变成1
    pepper：椒噪声，随机将像素值变成0或-1，取决于矩阵的值是否带符号
    s&p：椒盐噪声
    speckle：均匀噪声（均值mean方差variance），out=image+n*image
seed： 可选的，int型，如果选择的话，在生成噪声前会先设置随机种子以避免伪随机
clip： 可选的，bool型，如果是True，在添加均值，泊松以及高斯噪声后，会将图片的数据裁剪到合适范围内。如果谁False，则输出矩阵的值可能会超出[-1,1]
mean： 可选的，float型，高斯噪声和均值噪声中的mean参数，默认值=0
var：  可选的，float型，高斯噪声和均值噪声中的方差，默认值=0.01（注：不是标准差）
local_vars：可选的，ndarry型，用于定义每个像素点的局部方差，在localvar中使用
amount： 可选的，float型，是椒盐噪声所占比例，默认值=0.05
salt_vs_pepper：可选的，float型，椒盐噪声中椒盐比例，值越大表示盐噪声越多，默认值=0.5，即椒盐等量
--------
返回值：ndarry型，且值在[0,1]或者[-1,1]之间，取决于是否是有符号数
'''

img = cv.imread("lenna.png")
noise_gs_img = util.random_noise(img, mode='speckle', var=0.1)

cv.imshow("source", img)
cv.imshow("lenna", noise_gs_img)
# cv.imwrite('lenna_noise.png',noise_gs_img)
cv.waitKey(0)
cv.destroyAllWindows()
```

