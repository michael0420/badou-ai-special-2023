# Assignment for Week 9
## numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
在指定的间隔内返回均匀间隔的数字。在[start, stop]这个区间的端点可以任意的被排除在外，默认包含端点；retstep=True时，显示间隔长度。  
## numpy.newaxis None的方便别名，对于索引数组是有用的。
```python
>>> np.newaxis is None
True
>>> x = np.array([1,2,3]) # 一维数组[1,2,3]
>>> x
array([1, 2, 3])
>>> x.shape
(3,)
>>> x[:,np.newaxis]# 二维数组
array([[1],
       [2],
       [3]])
>>> x[:,np.newaxis].shape
(3, 1)
>>> x[np.newaxis,:]# 二维数组
array([[1, 2, 3]])
>>> x[np.newaxis,:].shape
(1, 3)
>>> x[:,np.newaxis,np.newaxis]# 三维数组
array([[[1]],
 
       [[2]],
 
       [[3]]])
>>> x[:,np.newaxis,np.newaxis].shape
(3, 1, 1)
```
## np.random.normal(loc,scale,size）
* np.random.normal(loc,scale,size）
参数说明：
loc:正太分布的均值
scale:正太分布的标准差
size:设定数组形状
