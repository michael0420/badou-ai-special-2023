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
## torch.rand()和torch.randn()的区别
1. torch.rand()：
torch.rand() 用于生成元素值在 [0, 1) 之间均匀分布的随机张量。
返回的张量中的每个元素都是从区间 [0, 1) 的均匀分布中随机采样得到的。
示例：
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
