### hist_color
��ɫֱ��ͼ�Ա�

<img src="picture\image-20231102230131728.png" alt="image-20231102230131728" style="zoom:50%;" />

### hist_gray
�Ҷ�ֱ��ͼ�Ա�

<img src="picture\image-20231102230208232.png" alt="image-20231102230208232" style="zoom:50%;" />


### gaussian_noise  
��˹���� ˼·�� ȡ���е�İٷֱȵ������ĵ㣬����������ص��޸�

<img src="picture\image-20231102230456976.png" alt="image-20231102230456976" style="zoom:50%;" />



### ��������

�������� ˼·�� ȡ���е�İٷֱȵ������ĵ㣬����������ص��޸� ������0����255��

<img src="picture\image-20231102230332105.png" alt="image-20231102230332105" style="zoom: 50%;" />

### ��������

<img src="picture\image-20231102230731466.png" alt="image-20231102230731466" style="zoom:50%;" />

### ��������ʹ�÷���

```
import cv2 as cv
import numpy as np
from PIL import Image
from skimage import util

'''
def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
���ܣ�Ϊ������ͼƬ��Ӹ����������
������
image������ͼƬ�����ᱻת���ɸ����ͣ���ndarray��
mode�� ��ѡ��str�ͣ���ʾҪ��ӵ���������
    gaussian����˹����
    localvar����˹�ֲ��ļ����������ڡ�ͼ�񡱵�ÿ���㴦����ָ���ľֲ����
    poisson����������
    salt�������������������ֵ���1
    pepper�������������������ֵ���0��-1��ȡ���ھ����ֵ�Ƿ������
    s&p����������
    speckle��������������ֵmean����variance����out=image+n*image
seed�� ��ѡ�ģ�int�ͣ����ѡ��Ļ�������������ǰ����������������Ա���α���
clip�� ��ѡ�ģ�bool�ͣ������True������Ӿ�ֵ�������Լ���˹�����󣬻ὫͼƬ�����ݲü������ʷ�Χ�ڡ����˭False������������ֵ���ܻᳬ��[-1,1]
mean�� ��ѡ�ģ�float�ͣ���˹�����;�ֵ�����е�mean������Ĭ��ֵ=0
var��  ��ѡ�ģ�float�ͣ���˹�����;�ֵ�����еķ��Ĭ��ֵ=0.01��ע�����Ǳ�׼�
local_vars����ѡ�ģ�ndarry�ͣ����ڶ���ÿ�����ص�ľֲ������localvar��ʹ��
amount�� ��ѡ�ģ�float�ͣ��ǽ���������ռ������Ĭ��ֵ=0.05
salt_vs_pepper����ѡ�ģ�float�ͣ����������н��α�����ֵԽ���ʾ������Խ�࣬Ĭ��ֵ=0.5�������ε���
--------
����ֵ��ndarry�ͣ���ֵ��[0,1]����[-1,1]֮�䣬ȡ�����Ƿ����з�����
'''

img = cv.imread("lenna.png")
noise_gs_img = util.random_noise(img, mode='speckle', var=0.1)

cv.imshow("source", img)
cv.imshow("lenna", noise_gs_img)
# cv.imwrite('lenna_noise.png',noise_gs_img)
cv.waitKey(0)
cv.destroyAllWindows()
```

