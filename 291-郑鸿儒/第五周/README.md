## 透视变换与图像聚类算法

#### 透视变换
    1. 基本思想
        寻找四个点， 对应求出变换矩阵，将原图每个点都与变换矩阵相乘得到目标图像
    2. 接口
        cv2.getPerspectiveTransform(src, dst)  ==> 求解变换矩阵
        cv2.warpPerspectiveMatrix(src, matrix, dstSize) ==> 求变换后的图像
    3. 求变换后的图像(手动实现)
        1. 根据目标图像大小，遍历每个目标图像像素点
        2. 定义矩阵齐次坐标矩阵[x, y, 1](即[width, height, 1])
        3. 变换矩阵的逆阵左乘齐次坐标矩阵
        4. 相乘后res[0] / res[2], res[1] / res[2]即可得到当前点对应原图中的位置（虚拟点）
        5. 使用插值算法获得虚拟点像素值，赋值给当前目标点
        ps: 由于手动实现的误差，插值算法选择直接int或round消除小数部分效果反而更好
    4. 顶点检测
        1. (可选)高斯滤波减少噪声影响
        2. cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))) 膨胀操作，扩展高亮部分(白色区域)
        3. cv2.Canny 检测边缘，输出所有边缘的二值图
        4. cv2.findContours(edge, cv2.RETR_EXTERNSL. cv2.CHAIN_APPROX_SIMPLE) 检测所有轮廓
        5. 对轮廓进行诸如排序的操作，选择满足条件的轮廓
        6. cv2.approxPolyDP(cnt, epsilon, closed) 拟合多边形，epsilon: 经验值，越小拟合越精确，一般选周长的1%~5%， 返回端点坐标


#### 聚类算法
    1. K均值聚类(K-Means)算法
        1. 基本思想
            选取几个点作为质心，不断重复计算其余点到质心的距离，并重新选取质心计算距离，重复多次知道满足停止条件(精度或迭代次数)
        2. 接口
            cv2.kmeans(src, k, bestLabels, criteria, attempts, flags)
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k)
            kmeans.fit_predict(data)
        3. RGB
            对于RGB的k均值聚类，可将图像矩阵reshape为(n, 3)的矩阵，每一行的三个通道作为一个样本， 对此矩阵进行k均值聚类，根据结果标签选取适当的数据中心替换所有的rgb值
    2. 层次聚类
        1. 基本思想
            将每个数据作为一类，找到距离最近的两类进行合并，多次合并最终合为一类
        2. 接口
            from scipy.cluster.hierachy import linkage, fcluster, dendrogram
            linkage(data, method): 计算样本间距离及相似性，并执行合并类的操作，返回层次聚类的链接矩阵
            fcluster(z, t, criterion): 根据给定阈值对聚类结果进行切割，并返回每个样本所属聚类的标签
            dendrogram(z, p): 根据聚类结果绘制树状图 (p: 指定要显示的叶节点的数量)
    3. 密度聚类
        1. 基本思想
            设定参数ε，min_sample, 对每个未访问点计算ε邻域，邻域内点的数量大于等于min_sample 则视为一类
        2. 接口
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=0.5, min_sample)
            dbscan.fit(data) 返回分类标签
        
