# Assignment for Week 4
## np.mean(data, axis=0)用法
### 代码
```python
import numpy as np
X = np.array([[1, 2], [4, 5], [7, 8]])
print np.mean(X, axis=0, keepdims=True)
print np.mean(X, axis=1, keepdims=True)
```
### 输出结果
```python
                 [[ 1.5]
 [[ 4.  5.]]      [ 4.5]    
                  [ 7.5]]
```
**axis=0，那么输出矩阵是1行，求每一列的平均（按照每一行去求平均）；**  
**axis=1，输出矩阵是1列，求每一行的平均（按照每一列去求平均），**  
**还可以这么理解，axis是几，那就表明哪一维度被压缩成1。**  

## 非极大值抑制（Non-Maximum Suppression，NMS）
**消除冗余**  
**其思想是搜素局部最大值，抑制非极大值。**  

## np.pad(array, pad_width, mode, **kwargs)用法
### 参数  
array——表示需要填充的数组；  
pad_width——表示每个轴（axis）边缘需要填充的数值数目。
参数输入方式为：（(before_1, after_1), … (before_N, after_N)），其中(before_1, after_1)表示第1轴两边缘分别填充before_1个和after_1个数值。取值为：{sequence, array_like, int}  
mode——表示填充的方式（取值：str字符串或用户提供的函数）,总共有11种填充模式    
### 填充方式  
‘constant’——表示连续填充相同的值，每个轴可以分别指定填充值，constant_values=（x, y）时前面用x填充，后面用y填充，缺省值填充0  
‘edge’——表示用边缘值填充  
‘linear_ramp’——表示用边缘递减的方式填充  
‘maximum’——表示最大值填充  
‘mean’——表示均值填充  
‘median’——表示中位数填充  
‘minimum’——表示最小值填充  
‘reflect’——表示对称填充  
‘symmetric’——表示对称填充  
‘wrap’——表示用原数组后面的值填充前面，前面的值填充后面  
