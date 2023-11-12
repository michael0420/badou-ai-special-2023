## 向量

### 什么是向量？

向量是具有大小和方向的量，通常用箭头表示其方向和长度表示其大小。

### 向量表达方式

平面直角坐标系也常被用来表示向量。在这种表示法中，我们选取与 x 轴和 y 轴方向相同的两个单位向量 i 和 j 作为基底。然后，以原点 O 为起点、P 为终点作向量 a。
此外，对于更复杂的向量，例如三维向量，我们可以使用矩阵来进行表示。例如，三维向量 a 可以表示为矩阵形式：$\vec {a} = \begin {bmatrix} x \\ y \\ z \end {bmatrix}$
同时，我们也有专门的数学运算来计算向量的大小，也就是向量的模。
如果向量 a 的坐标为(x,y,z)，那么它的模长可以通过公式来计算：
$\left | \vec {a} \right | = \sqrt {x^2 + y^2+ z^2}$

## 线性变换

### 什么是线性变换？

线性变换，从字面上理解，就是保持向量加法和数乘的线性性质的变换。
它可以被看作是一种特殊的函数，将一个向量空间里的向量映射到另一个向量空间里的另一个向量。
具体来说，假设有一个对向量的线性变换叫做 T，那么按照线性变换的性质，我们有$T(v+w)=T(v)+T(w)$，$T(c*v)=c*T(v)$对所有 c 都成立。这意味着线性变换保持了向量的加法和数乘的运算结构。
实际上，常见的线性变换有恒等变换（单位变换）、求微商、求定积分等。
例如，我们可以把一束光打在一个正方体上，在桌面上形成一个影子，那么影子与正方体之间就相当于存在一个线性变换，三维到二维的线性变换。

## 矩阵

### 矩阵是什么？

矩阵是一种数学对象，可以看作是一个由数排成行和列的矩形阵列。我们通常使用大写字母表示矩阵，例如$\mathbf{A} 、\mathbf{B}、\mathbf{C}$等。元素可以是任何数值、变量、表达式或函数。

### 矩阵的类型

按照矩阵的行列情况，我们可以将矩阵分为行矩阵、列矩阵、零矩阵以及$n$阶方阵等类型。具体来说，$m \times n$阶矩阵中，如果$m=1$，就被称为行矩阵，或者$n$维行向量；同样地，如果$n=1$，就被称为列矩阵，或者$m$维列向量。而所有元素都为 0 的$m\times n$阶矩阵被称为零矩阵。

#### 单位矩阵

单位矩阵是一个$n \times m$矩阵，从左到右的对角线上的元素是 1，其余元素都为 0。
如：$$\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$ $$\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
如果$\mathbf{A}$是$m \times n$矩阵，$\mathbf{I}$是单位矩阵，则$$\mathbf{AI} = \mathbf{A}, \mathbf{IA} = \mathbf{A},$$

$$
\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \times
\begin{bmatrix} a & b \\ c & d \end{bmatrix}
=
\begin{bmatrix}
1 \times a + 0 \times c & 1 \times b + 0 \times d
\\
0 \times a + 1 \times c & 0 \times b + 1 \times d
\end{bmatrix}
=
\begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$

$$
\begin{bmatrix} a & b \\ c & d \end{bmatrix} \times
\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
=
\begin{bmatrix}
a \times 1 + b \times 0 & a \times 0 + b \times 1
\\
c \times 1 + d \times 0 & c \times 0 + d \times 1
\end{bmatrix}
=
\begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$

> 单位矩阵在矩阵乘法中的作用相当于数字`1`

#### 逆矩阵

逆矩阵$\mathbf{A^{-1}}$
逆矩阵是矩阵理论的重要概念，如果一个 n 阶方阵 A 在相同的数域上存在另一个 n 阶矩阵 B，使得：$AB=BA=E，则我们称B是A的逆矩阵，而A被称为可逆矩阵。其中，E为单位矩阵。$
然而，并非所有矩阵都存在逆矩阵。首先，矩阵必须是一个方阵。其次，矩阵的行列式不能为 0。

##### A 的逆矩阵

$\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} 的逆矩阵$

$\lvert \mathbf{A} \rvert
ad - bc
, \lvert \mathbf{A} \rvert 是\mathbf{A}的二阶行列式$

$\mathbf{A^{-1}}=
\cfrac{1}{\lvert \mathbf{A} \rvert}
\begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}$

##### B 的逆矩阵

$\mathbf{B} =
\begin{bmatrix}
3 & -4 \\
2 & -5
\end{bmatrix} 的逆矩阵$

$$
\mathbf{B^{-1}}=
\cfrac{1}{\lvert \mathbf{B} \rvert}
\begin{bmatrix}
-5 & 4 \\
-2 & 3
\end{bmatrix}
=
\cfrac{1}{(-5) \times 3 - (-4) \times 2}
\begin{bmatrix}
-5 & 4 \\
-2 & 3
\end{bmatrix}
=
-\cfrac{1}{7}
\begin{bmatrix}
-5 & 4 \\
-2 & 3
\end{bmatrix}
=
\begin{bmatrix}
\frac{5}{7} & -\frac{4}{7} \\
\frac{2}{7} & -\frac{3}{7}
\end{bmatrix}
$$

计算逆矩阵的方法有很多，比如初等行运算（高斯－若尔当）。
在我们介绍逆矩阵的计算方法之前，需要先明确一点，逆矩阵不等于矩阵转置。矩阵转置的操作是将一个矩阵行和列互换，在线性代数当中，矩阵 A 的转置记作$\mathbf{A^T}$，而 A 的逆矩阵记作$\mathbf{A^{-1}}$，虽然看起来比较相似，但二者是有本质区别的。

#### 奇异矩阵

**当一个矩阵没有逆矩阵的时候，称该矩阵为奇异矩阵。**这是因为在数学中，如果一个矩阵的行列式为零，那么这个矩阵就被称为奇异矩阵或者非可逆矩阵。
具体来说，对于一个$n$阶方阵$A$，如果它的行列式等于零，那么根据线性代数的基本定理，能够知道 A 没有逆矩阵。换句话说，对于任何向量$x$，我们都有$Ax=0$，这意味着$x$是$A$的一个解，但$A$并没有唯一确定的解。因此，$A$被认为是奇异的，或者说是不可逆的。

$$
\begin{align*}
A \in \mathbb{R}^{n \times n} \\
n\text{det}(A) = 0 \\
n\Rightarrow A^{-1} \notin \mathbb{R}^{n \times n} \\
n\Rightarrow A \text{ is singular or non-invertible.}
\end{align*}
$$

其中，$\mathbb{R}^{n \times n}$表示实数空间中的 n 阶方阵，$\text{det}(A)$表示矩阵 A 的行列式，$A^{-1}$表示矩阵 A 的逆矩阵，$\notin$表示不等于。

### 矩阵的表达

在矩阵的表达方式上，我们需要遵循以下规则：矩阵元素必须在"[]"内；同行元素之间用空格（或","）隔开；行与行之间用";"（或回车符）隔开。

### 矩阵基本运算

矩阵的基本运算包括加法、减法、乘法和标量乘法。

#### 矩阵加减法

两个同型矩阵相加或相减，结果仍为同型矩阵。设$A$和$B$是同型矩阵，则它们的和$C=A+B$和差$D=A-B$分别满足以下条件：
$C_{ij} = A_{ij} + B_{ij} \qquad D_{ij} = A_{ij} - B_{ij}$
其中，$C_{ij}$和$D_{ij}$分别表示$C$和$D$中第$i$行第$j$列的元素。

运算方式：

- 矩阵加法：$\mathbf{C} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} + \begin{bmatrix} e & f \\ g & h \end{bmatrix} = \begin{bmatrix} a+e & b+f \\ c+g & d+h \end{bmatrix}$
- 矩阵减法：$\mathbf{D} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} - \begin{bmatrix} e & f \\ g & h \end{bmatrix} = \begin{bmatrix} a-e & b-f \\ c-g & d-h \end{bmatrix}$

#### 标量乘法

一个矩阵和一个标量相乘，结果是将该标量乘以该矩阵的所有元素。设$\mathbf{A}$是一个矩阵，$\lambda$是一个标量，则它们的乘积$\mathbf{A}\lambda$满足以下条件：
$(\mathbf{A}\lambda)_{ij} = \lambda A_{ij}$
其中，$\mathbf{A}\lambda$表示将$\mathbf{A}$的所有元素都乘以$\lambda$,而$\lambda A_{ij}$表示将$\lambda$乘以$\mathbf{A}$的第$i$行第$j$列的元素。

运算方式：

- 标量乘法：$\mathbf{A}\lambda = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \lambda = \begin{bmatrix} a\lambda & b\lambda \\ c\lambda & d\lambda \end{bmatrix}$

#### 矩阵乘法

两个矩阵相乘，结果是一个积矩阵。设$A$和$B$是两个$m \times n$矩阵和$n \times p$矩阵，则它们的积$C=AB$满足以下条件：
$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$
其中，$C_{ij}$表示$C$中第$i$行第$j$列的元素，而$\sum_{k=1}^{n} A_{ik} B_{kj}$表示对$B$的第$j$列元素求和，并对每个元素与对应的$A$中第$i$行元素相乘再求和。

运算方式：

- 矩阵乘法：$\mathbf{C} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} e & f \\ g & h \end{bmatrix} = \begin{bmatrix} ae + bg & af + bh \\ ce + dg & cf + dh \end{bmatrix}$

## 导数&偏导数

### 导数是什么？

导数(微分)：是**代表函数(曲线)的斜率**，是描述函数(曲线)变化快慢的量，同时曲线的极大值点也可以使用导数来判断，即极大值点的导数为 0，此时斜率为零。
偏导数：是指在多元函数的情况下，对其每个变量进行求导，求导时，把其他变量看做常量进行处理，物理意义就是查看这一个变量在其他情况不变的情况下对函数的影响程度。

### 导数的表达方式：

基本定义：$f'(x) = \lim_{\Delta x \to 0} \frac{\Delta y}{\Delta x}
=\lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$
一阶导数可以表示为$\frac{\partial f}{\partial x}$，二阶导数可以表示为$\frac{d^2}{dx^2} f(x)$。
对于偏导数，例如函数$f(x, y)$对$x$的偏导数，可以写为$\frac{\partial}{\partial x} f(x, y)$。

## 梯度

### 梯度的介绍

梯度是微积分中的一个重要概念，它描述了一个向量场在某一点的方向导数沿着该方向取得最大值，即函数在该点处沿着此方向（梯度的方向）变化最快，变化率最大（为梯度的模）。

简而言之，对多元函数的各个自变量求偏导数，并把求得的这些偏导数写成向量形式，就是梯度。
梯度的基本表达式是：
$\nabla f(x, y, z, ...) = 
\left(\frac{\partial f}{\partial x}, 
\frac{\partial f}{\partial y}, 
\frac{\partial f}{\partial z}, 
...
\right)$
其中，$\frac{\partial f}{\partial x}$、$\frac{\partial f}{\partial y}$、$\frac{\partial f}{\partial z}$ 等表示函数 $f$ 关于变量$x$、$y$、$z$的偏导数。

梯度通常用于机器学习和深度学习中的优化算法，如梯度下降法。

### 基本思想

梯度的基本思想是通过计算函数在给定点处的梯度来找到函数在该点处最陡峭上升的方向，从而确定函数在该点处的最佳搜索方向。**梯度是一个向量**，它的**方向是函数增长最快的方向**，而它的**模是函数在该方向上的增长速率**。

### 原理

梯度的原理是基于微积分中的偏导数。对于一个多变量函数 (f(x, y, z, ...))，其梯度是一个向量，包含所有偏导数。梯度的方向是函数增长最快的方向，而它的模是函数在该方向上的增长速率。

### 作用

梯度的作用主要有以下几点：

1. 梯度可以用于**优化问题**。例如，在机器学习中，我们经常需要最小化损失函数来训练模型。通过计算损失函数关于参数的梯度，我们可以使用梯度下降法等优化算法来更新参数，从而最小化损失函数。
2. 梯度可以帮助我们理解复杂函数的性质。例如，通过计算一个复杂函数的梯度，我们可以了解该函数在不同位置的变化情况。

### 梯度下降算法

#### 基本思路

...

#### 主要原理

1. 确定小目标（预测函数）

   _机器学习常见的任务是通过学习算法，自动发现数据背后的规律不断改进模型并做出预测。_
   基本方法：

1. 找一个过原点的直线$y = wx$，即预测函数
1. 然后计算所有样本点与直线的偏离程度
1. 根据误差大小来调整斜率$w$

1. 找到差距（代价函数）

   ![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697969570364-b8e41ae6-05af-42be-b9a9-8caa4749051c.png#averageHue=%23030302&clientId=u94a33e38-13fa-4&from=paste&height=389&id=u24146f77&originHeight=778&originWidth=1294&originalType=binary&ratio=2&rotation=0&showTitle=false&size=187382&status=done&style=none&taskId=u42749781-a5da-499e-be4f-a80a0356f05&title=&width=647)![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697969545172-2df16213-541e-4404-8666-3a0166f11dfc.png#averageHue=%23050505&clientId=u94a33e38-13fa-4&from=paste&height=400&id=ub7fdb3e6&originHeight=800&originWidth=1254&originalType=binary&ratio=2&rotation=0&showTitle=false&size=135004&status=done&style=none&taskId=u0f0c3da8-e1e1-4974-b8a1-c37b23501f9&title=&width=627)
   量化数据的偏离程度，即误差（常见的量化方式有：均方误差【误差平方和的平均值】）

- 得到误差函数，它代表了学习所需要付出的代价，故常被称为代价函数

通过定义预测函数，然后根据误差公式推导代价函数，成功将样本点拟合过程映射到了函数

3. 明确搜索方向（梯度计算）

   ![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697969453959-7b1189bf-274c-4506-b91b-0db7264d1c88.png#averageHue=%23040303&clientId=u94a33e38-13fa-4&from=paste&height=327&id=u48085d58&originHeight=654&originWidth=1442&originalType=binary&ratio=2&rotation=0&showTitle=false&size=192301&status=done&style=none&taskId=u2cfc53af-ca3a-405e-a5ab-0a0f8202387&title=&width=721)

4. 大胆的往前走吗？（学习率）

   ![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697969453959-7b1189bf-274c-4506-b91b-0db7264d1c88.png#averageHue=%23040303&clientId=u94a33e38-13fa-4&from=paste&height=327&id=RbNGK&originHeight=654&originWidth=1442&originalType=binary&ratio=2&rotation=0&showTitle=false&size=192301&status=done&style=none&taskId=u2cfc53af-ca3a-405e-a5ab-0a0f8202387&title=&width=721)
   ![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697969669972-58d1e191-9643-4b8b-9126-c9a8d73fab2a.png#averageHue=%23030202&clientId=u94a33e38-13fa-4&from=paste&height=336&id=ua807bbb2&originHeight=672&originWidth=1462&originalType=binary&ratio=2&rotation=0&showTitle=false&size=160295&status=done&style=none&taskId=u7d05ffe0-8cbf-4c4a-9dc3-2be549c8f10&title=&width=731)

5. 不达目的不罢休（循环迭代）

   循环计算梯度和按学习率前进这两步，直到找到最低点。
   ![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697969453959-7b1189bf-274c-4506-b91b-0db7264d1c88.png#averageHue=%23040303&clientId=u94a33e38-13fa-4&from=paste&height=327&id=BgXXy&originHeight=654&originWidth=1442&originalType=binary&ratio=2&rotation=0&showTitle=false&size=192301&status=done&style=none&taskId=u2cfc53af-ca3a-405e-a5ab-0a0f8202387&title=&width=721)
   ![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697969669972-58d1e191-9643-4b8b-9126-c9a8d73fab2a.png#averageHue=%23030202&clientId=u94a33e38-13fa-4&from=paste&height=336&id=pf1Wj&originHeight=672&originWidth=1462&originalType=binary&ratio=2&rotation=0&showTitle=false&size=160295&status=done&style=none&taskId=u7d05ffe0-8cbf-4c4a-9dc3-2be549c8f10&title=&width=731)

#### 没这么简单

在实际工作中，训练样本千奇百怪，代价函数千变万化，不太可能是简单的抛物线
如：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697973161844-2cdb3d23-6c25-45bd-a8f9-66a970f8b237.png#averageHue=%23838481&clientId=u94a33e38-13fa-4&from=paste&height=381&id=ue3f2ce61&originHeight=762&originWidth=1296&originalType=binary&ratio=2&rotation=0&showTitle=false&size=589134&status=done&style=none&taskId=ubfb1b29c-df8a-4c31-a9d6-0ef73433807&title=&width=648)
![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697973143029-beb32951-853e-4d5a-b74a-f7365d43ea7b.png#averageHue=%23020201&clientId=u94a33e38-13fa-4&from=paste&height=343&id=u313f96c2&originHeight=686&originWidth=1372&originalType=binary&ratio=2&rotation=0&showTitle=false&size=220602&status=done&style=none&taskId=ua2cc258d-8340-4551-b5b6-6075ffd2229&title=&width=686)
![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697973207237-e4d8b9cd-1778-4f30-b301-9c807209b43e.png#averageHue=%2326211a&clientId=u94a33e38-13fa-4&from=paste&height=427&id=ud01b3600&originHeight=854&originWidth=1424&originalType=binary&ratio=2&rotation=0&showTitle=false&size=680364&status=done&style=none&taskId=u0d8e2cfc-4ee3-482a-a41a-f98c86e448c&title=&width=712)

#### 梯度下降法的各种变体

##### Batch Gradient Descent(BGD)

![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697973288402-2d533501-3993-4cb4-8594-63dd581078c3.png#averageHue=%23060604&clientId=u94a33e38-13fa-4&from=paste&height=375&id=ud6236378&originHeight=750&originWidth=1374&originalType=binary&ratio=2&rotation=0&showTitle=false&size=438964&status=done&style=none&taskId=u9efc2bf4-94db-4cdc-85ed-ca1ea032dd2&title=&width=687)
特点：全部训练样本参与训练
优点：保证算法精确度，找到全局最优点
缺点：计算速度慢

##### Stochastic Gradient Descent(SGD)

![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697973502319-67f4fa09-aee9-4c09-8bc1-eaa438aacb1a.png#averageHue=%23080707&clientId=u94a33e38-13fa-4&from=paste&height=408&id=ued8b5ed9&originHeight=816&originWidth=1350&originalType=binary&ratio=2&rotation=0&showTitle=false&size=423279&status=done&style=none&taskId=ubb1adbdb-3c39-44c5-9c14-1eda4c00981&title=&width=675)
特点：每下降一步只需一个样本参与计算
优点：速度快
缺点：精准度较差

##### MBGD

![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697974423051-ebd3cf51-c799-49b9-88c8-0e79f60ec355.png#averageHue=%23050504&clientId=u94a33e38-13fa-4&from=paste&height=368&id=ub941dd46&originHeight=736&originWidth=1416&originalType=binary&ratio=2&rotation=0&showTitle=false&size=434380&status=done&style=none&taskId=u91c81492-1f5b-4c9a-8389-ce1ad7a360a&title=&width=708)
特点：每下降一步只需一批次样本参与计算
优点：速度较快的同时保证了一定的精准度

##### 其他更优的算法：

- Adagrad 动态调节学习率：不常更新的学习率增大，频繁更新的学习率降低
  - 问题是：频繁跟新的学习率过小，以致逐渐消失
- RMSProp 优化动态调节学习率：解决 Adagrad 不足之处
- AdaDelta：无需设置学习率
- Adam：融合 AdaGrad 和 RMSProp
- Momentum：模拟动量
  - 下降过程中，充分考虑前一阶段下降的惯性
- FTRL……

#### 梯度下降算法并非完美无缺

- 对于学习率的设定十分敏感；太大会反复横跳，太小会浪费计算力
- 除 BGD 外，无法保证找到全局最低点，可能会陷入局部低点难以自拔

## 概率学基础

### Machine Learning 与 Traditional statistical analyses

- **关注主体方面**，传统统计分析主要关注样本数据的特性、分布和相关性等，通过统计模型来解释和预测数据。而机器学习则更加关注从大量数据中学习到的模型的泛化能力和准确性。
- **模型复杂度方面**，传统统计分析倾向于选择较为简单的模型，以降低过拟合的风险和解释结果的困难性。而机器学习则通常允许使用更复杂的模型，以更好地捕捉数据中的模式和特征。
- **验证性方面**，传统统计分析更加注重对模型参数的假设检验和置信区间估计，以及模型的显著性和稳定性等方面的验证。而机器学习则更多地关注于使用交叉验证、留出法等技术来评估模型的性能和泛化能力。

然而，需要注意的是，并非所有的情况下都可以简单地将这两者进行划分。在实践中，我们往往会结合传统的统计分析方法和机器学习算法来解决问题。例如，可以使用传统统计分析方法对数据进行预处理和探索性分析，然后利用机器学习算法构建复杂模型来进行预测和决策。
综上所述，虽然 Machine Learning 和 Traditional Statistical Analyses 在关注主体和验证性等方面存在一些区别，但这并不是绝对的界限。在实践中，我们需要根据问题的特点和需求灵活地选择合适的方法和策略。

### 事件基础关系

概率学基础中的事件是指一个或多个基本事件的集合，而关系运算则是指对事件进行组合、交、并等操作。
常见的关系运算包括：

1.  并集（Union）：两个事件 A 和 B 的并集表示为$A∪B$，包含 A 和 B 中的所有基本事件。
2.  交集（Intersection）：两个事件 A 和 B 的交集表示为$A∩B$，包含同时属于 A 和 B 的基本事件。
3.  补集（Complement）：一个事件 A 的补集表示为$A'$，包含所有不属于 A 的基本事件。
4.  差集（Difference）：两个事件 A 和 B 的差集表示为$A-B$，包含属于 A 但不属于 B 的基本事件。

### 运算律

这些关系运算满足以下运算律：

1.  交换律：对于任意事件 A 和 B，有$A∪B = B∪A$和$A∩B = B∩A$。
2.  结合律：对于任意事件 A、B 和 C，有$(A∪B)∪C = A∪(B∪C)$和$(A∩B)∩C = A∩(B∩C)$。
3.  分配律：对于任意事件 A、B 和 C，有$A∪(B∩C) = (A∪B)∩(A∪C)$和$A∩(B∪C) = (A∩B)∪(A∩C)$。

_这些运算律可以通过直观的方式来理解：_

1. _ 交换律：将两个事件的并集或交集的顺序颠倒，结果不变。例如，对于两个事件 A 和 B，无论是先计算 A∪B 还是先计算 B∪A，得到的结果都是相同的。同样地，无论是先计算 A∩B 还是先计算 B∩A，得到的结果也是相同的。 _
2. _ 结合律：将两个事件的并集或交集与第三个事件进行组合时，可以先将前两个事件的并集或交集进行组合，再与第三个事件进行组合，或者先将第三个事件与前两个事件分别进行组合，再将结果进行组合。这两种方式得到的结果是一致的。例如，对于三个事件 A、B 和 C，无论是先计算(A∪B)∪C 还是先计算 A∪(B∪C)，得到的结果都是相同的。同样地，无论是先计算(A∩B)∩C 还是先计算 A∩(B∩C)，得到的结果也是相同的。 _
3. _ 分配律：将一个事件的并集或交集与另一个事件的差集进行组合时，可以先将第一个事件的并集或交集与第二个事件进行组合，再从结果中去掉属于第三个事件的基本事件；或者先将第一个事件与第二个事件的差集进行组合，再与第三个事件进行组合。这两种方式得到的结果是一致的。例如，对于三个事件 A、B 和 C，无论是先计算 A∪(B∩C)还是先计算(A∪B)∩(A∪C)，得到的结果都是相同的。同样地，无论是先计算 A∩(B∪C)还是先计算(A∩B)∪(A∩C)，得到的结果也是相同的。 _

### 概率基本概念

概率是描述随机事件发生可能性的数学工具。在概率论中，我们使用概率来量化不确定性。概率的范围在 0 到 1 之间，其中 0 表示不可能事件，1 表示必然事件。

#### 基本性质

关于概率的性质有很多，以下是一些主要的性质：

- 非负性：对于任意事件 A，$P(A) ≥ 0$；
- 规范性：所有可能的事件 A 的集合对应的概率值为 1；
- 可列可加性：若事件 A1，A2，…，An 两两互不相容，则其和事件$A1+A2+…+An$的概率为各事件概率之和；
- 交换性：对于任意两个事件 A 和 B，都有$P(A∩B)=P(A)P(B)$；
- 增减性：若事件 A 发生导致事件 B 也发生，则必有$P(A∪B)=P(A)+P(B)$；
- 幂等性：对于任意事件 A，都有$P(A^n)=P(A)^n（n为正整数）$。

#### 古典型概率

古典型概率是一种计算概率的方法，当随机事件中各种可能发生的结果及其出现的次数都可以由演绎或外推法得知，而无需经过任何统计试验即可计算各种可能发生结果的概率时，这种概率计算方法称为古典型概率或事前概率；比如掷硬币、掷骰子等试验，这些试验的结果是有限的且每个结果出现的概率是相同的，这种试验就叫做等可能概型，也叫作古典概型。
古典概型的计算公式是：$P (A)=\cfrac{N (A)}{N (Ω)}$，其中 N (A)表示事件 A 包含的基本事件个数，N (Ω)表示基本空间的基本事件的总个数。

#### 离散型随机变量

离散型随机变量是指其取值范围为有限或可数无限个的随机变量。例如，投掷一枚骰子的结果就是一个离散型随机变量，它的取值范围是{1,2,3,4,5,6}。

#### 连续型随机变量

连续型随机变量是指其取值范围为不可数无限的随机变量。例如，测量某个物体的长度就是一个连续型随机变量，它的取值范围可以是任意实数。

### 独立性

两个事件 A 和 B 独立，如果它们的发生不受彼此影响。用概率表示就是：$P(A∩B) = P(A)P(B)$。

### 离散

离散型随机变量的概率分布可以用概率质量函数（PMF）或概率质量函数（PDF）表示。
PMF 是一个映射，将每个可能的取值映射到一个非负实数，表示该取值发生的概率。
PDF 是一个累积分布函数，表示随机变量取值小于等于某个值的概率。

### 数学期望

在概率论和统计学中，数学期望（或均值，亦简称期望）是试验中每次可能结果的概率乘以其结果的总和，它是最基本的数学特征之一。离散型随机变量和连续型随机变量的期望值计算以及推导过程都有具体的公式和方法。

#### 计算公式

如果随机变量 X 的概率分布函数为 f(x)，则数学期望的计算公式为：
$E(X) = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$
对于离散型随机变量，数学期望值的计算公式为：
$E(X) = \sum_{x in X} x \cdot P(X=x)$
其中，P(X=x)表示随机变量取值 x 的概率。

### 方差

方差是在概率论和统计学中衡量随机变量或一组数据的离散程度的度量。它量化了随机变量或一组数据与它们的数学期望（即均值）之间的偏离程度。方差越小，代表这组数据越稳定，方差越大，代表这组数据越不稳定。

#### 计算公式

对于单个离散型随机变量$X$，其方差的计算公式为：
$Var(X) = E(X^2) - [E(X)]^2$
其中，$E(X^2)$表示随机变量$X$的平方的期望值，$[E(X)]^2$表示随机变量$X$的期望值的平方。

对于连续型随机变量$X$，其方差的计算公式为：
$Var(X) = \int_{-\infty}^{\infty} (x - E(X))^2 f(x) \, dx$
其中，$f(x)$为随机变量$X$的密度函数，$E(X)$为$X$的数学期望值。

在实际应用中，方差是一个重要的统计量，不仅用于描述数据分布的特性，还可用于推断和预测。

### 标准差

标准差，又被称为标准偏差或实验标准差，是广泛应用在概率统计中的一个重要概念，通常被用作衡量数据分布程度的重要依据。标准差能反映一个数据集的离散程度，即数据值之间的变异性或者分散性。需要注意的是，即使两个数据集的平均数相同，它们的标准差也可能不同。

对于单个连续型随机变量 X，其标准差的计算公式为：
$\sigma = \sqrt{\frac{(x_1 - \bar{x})^2 + (x_2 - \bar{x})^2 + ... + (x_n - \bar{x})^2}{n}}$
其中，$\bar{x}$表示随机变量 X 的算术平均值，$x_1, x_2, ..., x_n$表示随机变量 X 的取值，n 表示随机变量的取值个数。此外，样本标准差也可以通过方差的算术平方根得到。
总体标准差的公式与样本标准差的公式类似，不过其中的平均值是整个数据集的平均值。

### 正态分布（高斯分布）

正态分布，也被称为常态分布或高斯分布，是一个重要的连续概率分布。
如果随机变量 X 服从一个数学期望为 μ、方差为 σ^2 的正态分布，我们便记作$X \sim N(μ， σ^2)$。其概率密度函数呈钟形曲线，期望值 μ 决定了其位置，标准差 σ 决定了分布的幅度。
正态分布的概率密度函数（PDF）为：$f(x; μ， σ) = (\cfrac{1}{σ\sqrt{2π}})e^{-\cfrac{(x-μ)^2}{2σ^2}}$，其中 x 是随机变量的取值，μ 是均值，σ 是标准差。

正态分布有一个特殊的形态，即当 μ = 0, σ = 1 时的正态分布，称为标准正态分布，以 N(0, 1)表示。
标准正态分布的概率密度函数曲线关于直线 x=0 对称。

> _统计学家还制定了一张统计用表（自由度为 ∞ 时），借助该表就可以估计出某些特殊 u1 和 u2 值范围内的曲线下面积。此外，标准正态分布的面积（面积=比例=概率）在-1.96 ～+1.96 范围内曲线下的面积等于 0.9500，在-2.58 ～+2.58 范围内曲线下面积为 0.9900。_

正态分布在统计学和概率论中有广泛应用，例如描述自然现象、人类行为等。

## 熵（entropy）

物理学上，是“混乱” 程度的度量。
系统越有序，熵值越低；系统越混乱或分散，熵值越高。
信息理论:

1. 当系统的有序状态一致时，数据越集中的地方熵值越小，数据越分散的地方熵值越大。
   1. 这是从信息的完整性上进行的描述。
2. 当数据量一致时，系统越有序，熵值越低；系统越混乱或者分散，熵值越高。
   1. 这是从信息的有序性上进行的描述。

若不确定性越大，则信息量越大，熵越大
若不确定性越小，则信息量越小，熵越小

### 定义

假如事件 A 的分类划分是$(A1,A2,...,An)$，每部分发生的概率是$(p1,p2,...,pn)$，那信息熵定义为公式如下：
$Ent(A) = - \sum_{k=1}^nP_klog_2p_k$
