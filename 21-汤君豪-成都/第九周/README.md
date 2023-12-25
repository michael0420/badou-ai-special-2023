# Assignment for Week 9
## super(Net,self).__init__() 的含义
```python
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
```
python中的super(Net, self).init()

首先找到Net的父类（比如是类nn.Module），然后把类Net的对象self转换为类nn.Module的对象，然后“被转换”的类nn.Module对象调用自己的init函数

这是对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。

也就是说，子类继承了父类的所有属性和方法，父类属性自然会用父类方法来进行初始化。

当然，如果初始化的逻辑与父类的不同，不使用父类的方法，自己重新初始化也是可以的。比如：
```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
 
class Person(object):
    def __init__(self,name,gender,age):
        self.name = name
        self.gender = gender
        self.age = age
 
class Student(Person):
    def __init__(self,name,gender,age,school,score):
        #super(Student,self).__init__(name,gender,age)
        self.name = name.upper()  
        self.gender = gender.upper()
        self.school = school
        self.score = score
 
s = Student('Alice','female',18,'Middle school',87)
print s.school
print s.name
```
上面例子，父类对name和gender的初始化只是简单的赋值，但子类要求字母全部大写。
## enumerate用法总结
* enumerate()是python的内置函数
* enumerate在字典上是枚举、列举的意思
* 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
* enumerate多用于在for循环中得到计数  
例如对于一个seq，得到：
```python
(0, seq[0]), (1, seq[1]), (2, seq[2])
```
enumerate()返回的是一个enumerate对象  
如果对一个列表，既要遍历索引又要遍历元素时，首先可以这样写：
```python
list1 = ["这", "是", "一个", "测试"]
for i in range (len(list1)):
    print i ,list1[i]
```
上述方法有些累赘，利用enumerate()会更加直接和优美：  
```python
list1 = ["这", "是", "一个", "测试"]
for index, item in enumerate(list1):
    print index, item
>>>
0 这
1 是
2 一个
3 测试
```
enumerate还可以接收第二个参数，用于指定索引起始值，如：  
```python
list1 = ["这", "是", "一个", "测试"]
for index, item in enumerate(list1, 1):
    print index, item
>>>
1 这
2 是
3 一个
4 测试
```
**补充**  
如果要统计文件的行数，可以这样写：  
```python
count = len(open(filepath, 'r').readlines())
```
这种方法简单，但是可能比较慢，当文件比较大时甚至不能工作。  
可以利用enumerate()：  
```python
count = 0
for index, line in enumerate(open(filepath,'r'))： 
    count += 1
```
## torch.rand()和torch.randn()的区别
1. torch.rand()：  
torch.rand() 用于生成元素值在 [0, 1) 之间均匀分布的随机张量。  
返回的张量中的每个元素都是从区间 [0, 1) 的均匀分布中随机采样得到的。  
示例：  
```python
import torch
# 生成一个形状为 (2, 3) 的随机张量，值在 [0, 1) 的均匀分布中随机取样
x = torch.rand(2, 3)
```
2. torch.randn()：  
torch.randn() 用于生成元素值服从标准正态分布（均值为0，方差为1）的随机张量。  
返回的张量中的每个元素都是从标准正态分布中随机采样得到的。
```python
import torch
# 生成一个形状为 (2, 3) 的随机张量，值服从标准正态分布
x = torch.randn(2, 3)
```
总的来说，torch.rand() 生成的张量中的元素值来自 [0, 1) 的均匀分布，而 torch.randn() 生成的张量中的元素值来自标准正态分布。因此，你可以根据需要选择合适的随机初始化方法。  
## numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
在指定的间隔内返回均匀间隔的数字。在[start, stop]这个区间的端点可以任意的被排除在外，默认包含端点；retstep=True时，显示间隔长度。  
## numpy.newaxis
None的方便别名，对于索引数组是有用的。  
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
np.random.normal(loc,scale,size）参数说明：  
* loc:正太分布的均值
* scale:正太分布的标准差
* size:设定数组形状
