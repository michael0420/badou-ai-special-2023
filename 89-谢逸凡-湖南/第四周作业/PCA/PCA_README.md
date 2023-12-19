PCA的基础在于变换坐标系，我们希望投影后投影值尽可能地分散，而这种分散程度，在数学上可以用方差来表述。
$$
Var(a)=\frac1m\sum_{i=1}^m{(a_i-\mu)^2}
$$
但方差只是针对将二维降成一维的问题，对于更高纬度，我们需要考虑协方差的情况，当协方差为0的时候表示两个字段完全独立，为了使得协方差为0，我们选择第二个基时只能在第一个基方向上选择。因此最终选择的两个方向一定是正交的。

至此，我们得到了降维问题的优化目标：**将一组N维向量降为K维（K大于0，小于N），其目标是选择K个单位（模为1）正交基，使得原始数据变换到这组基上后，各字段两两间协方差为0，而字段的方差则尽可能大（在正交的约束下，取最大的K个方差）**。



### 协方差矩阵

假设只有a和b两个字段，那么我们将它们按行组成矩阵X:
$$
X=\begin{pmatrix}a_1&a_2&\cdots&a_m\\b_1&b_2&\cdots&b_m\end{pmatrix}
$$
然后用X乘以X的转置，并乘上系数1/m:
$$
\left.\frac1mXX^{\mathsf{T}}=\left(\begin{array}{ll}\frac1m\sum_{i=1}^ma_i^2&\frac1m\sum_{i=1}^ma_ib_i\\\frac1m\sum_{i=1}^ma_ib_i&\frac1m\sum_{i=1}^mb_i^2\end{array}\right.\right)
$$
将协方差矩阵对角化可以看出它们之间的关系：
设原始数据矩阵X对应的协方差矩阵为C，而P是一组基按行组成的矩阵，设Y=PX，则Y为X对P做基变换后的数据。设Y的协方差矩阵为D，我们推导一下D与C的关系：
$$
\begin{aligned}
\text{D}& =\quad\frac1mYY^{\mathsf{T}}  \\
&=\quad\frac1m(PX)(PX)^{\mathsf{T}} \\
&=\quad\frac1mPXX^\top P^\top  \\
&=\quad P(\frac1mXX^\mathsf{T})P^\mathsf{T}
\end{aligned}
$$
由上文知道，协方差矩阵C是一个是对称矩阵，在线性代数上，实对称矩阵有一系列非常好的性质：

1）实对称矩阵不同特征值对应的特征向量必然正交。

2）设特征向量λ重数为r，则必然存在r个线性无关的特征向量对应于λ，因此可以将这r个特征向量单位正交化。

## PCA算法

总结一下PCA的算法步骤：

设有m条n维数据。

1）将原始数据按列组成n行m列矩阵X

2）将X的每一行（代表一个属性字段）进行零均值化，即减去这一行的均值

3）求出协方差矩阵
$$
C=\frac1mXX^{\mathsf{T}}
$$
4）求出协方差矩阵的特征值及对应的特征向量

5）将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵P

6）Y=PX即为降维到k维后的数据



详细算法见博客[CodingLabs - PCA的数学原理](http://blog.codinglabs.org/articles/pca-tutorial.html)

