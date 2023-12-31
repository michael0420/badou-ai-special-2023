## SIFT(尺度不变特征变换)、 RANSAC、哈希

#### SIFT
    1. 主要流程
        1. 构建高斯金字塔
        2. 构建高斯差分金字塔DOG
        3. 差分金字塔的每个点是否是极大值
        4. 记录所有关键点
        5. 筛选关键点，使用阈值限制，去除不稳定点以及噪声(opencv 中使用contrastThreshold, 也可以使用类似Canny的算法，每个关键点的邻域必须也要是关键点)
        6. 关键点方向信息分配，将邻域分成若干份(通常是4 * 4)，每份中的每个像素点计算梯度，建立梯度方向直方图，选取数量最多的方向作为主方向，余下数量高于主方向80%的方向作为辅助方向(一般包括主方向在内选8个方向, 若高于主方向80%的方向少于7个，则从余下中选取最多的几个方向凑齐)，此时对于此关键点就有4 * 4 * 8 = 128维的描述符
        7. 将需要比对的两张图片中的关键点进行比对，可使用穷举法，将一张图中的每一个关键点都与另一张图的每一张关键点进行比较，若相似度较高(一般用欧氏距离远近，越近说明相似度越高)，则说明是同一位置
    2. 关键词
    尺度：计算及模拟人眼看远近物体的概念，关注整体则忽略部分细节(大尺度)， 关注局部则注重细节（小尺度）
    关键参数: 
        1. 金字塔部分
        检测关键点的数量s一般为3~5，以5为例，则需要s + 2 = 5层DOG，需要s + 3 = 6 层高斯金字塔，高斯金字塔中的平滑系数σ一般为1.6，系数k一般用2 ** (1 / s)
    3. 专利
    在opencv3.4.2 至4.3.0版本中使用cv2.xfeatures2D.SIFT_create()
    在新版本(4.4.0及以上)中，opencv提供了SIFT的改进升级版，不受专利限制，cv2.SIFT_create()调用
    4. 接口
    cv2.xfeatures2D.SIFT_create()  ver3.4.2 ~ 4.3.0
    cv2.SIFT_create()  ver4.4.0 ~
    sift.detetectAndCompute() 检测关键点及其描述，需事先创建sift对象

#### 最小二乘法
    1. 推导
    对残差平方和求k的偏导，求得k的表达式，回代线性回归公式，得b的表达式(见笔记)
    2. 评价
    最小二乘法基本绑定线性回归，且效率较低，对大规模的数据，或是方差较大的数据，计算量很大；同时对于误差较大的数据，会受噪声影响较大，从而得出与实际相差较大的结果

#### RANSAC(随机采样一致性)
    1. 简介
    RANSAC是一种思想而非一种算法，可以应用才很多领域，RANSAC不关心如何求得具体的模型，只关心如何取样以及不断提高取得内群数据求得更真实的模型的概率
    2. 基本流程
        1. 随机选取几个点作为内群数据
        2. 根据已知的数学模型，以及选取的内群点计算模型表达式
        3. 将其余的所有点全部回代入模型表达式中
        4. 选取合适的误差作为阈值，判断当前条件下有多少内群点
        5. 重复多次上述步骤至设定的停止条件(可以是迭代次数到达设定的最大值)
        6. 选择内群数量最多的模型作为最终的模型
    3. 参数
        对于RANSAC，若选取的点是真实内群的概率为w， 选取n个内群点全为真实内群点的概率w^n, 至少一个不是内群点的概率(1 - w^n), 若迭代k次成功的概率为
        p，则1 - p = (1 - w^n)^k => k = log(1 - p) / log(1 - w^n)
                                    p = 1 - (1 - w^n)^k
        n不变时，k越大，p越大，w不变时，n越大，所需的k就越大
        结论: n选取时，尽量小
    4. 评价
        优点: 鲁棒的估计参数，可以从含有大量据外电的数据中估计出高精度的参数
        缺点:
            1. 需要已知数学模型
            2. 迭代次数理论上无上限，有限次的迭代可能找到错误的结果
            3. 数据集中含有多个模型只能找到一个
            4. 需要设置与问题相关的阈值
            5. 若要找到高精度的模型，需要进行大量的迭代
