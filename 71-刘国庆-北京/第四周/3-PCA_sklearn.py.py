# 导入NumPy库，用于数组操作和数学计算
import numpy as np
# 从scikit-learn库的decomposition模块中导入PCA（主成分分析）类
from sklearn.decomposition import PCA

# 定义一个NumPy数组，包含6个样本，每个样本有4个特征
X = np.array(
    [[-1, 2, 66, -1],
     [-2, 6, 58, -1],
     [-3, 8, 45, -2],
     [1, 9, 36, 1],
     [2, 10, 62, 1],
     [3, 5, 83, 2]])
# 创建PCA对象，指定将数据降到2维
pca = PCA(n_components=2)
# 使用PCA对象对数据进行训练，学习数据的主成分
pca.fit(X)
# 将原始数据X进行降维操作，降维后的结果存储在newX变量中
newX = pca.fit_transform(X)
# 打印每个主成分所占的方差比例，用于衡量每个主成分的贡献程度
print(f"贡献度:{pca.explained_variance_ratio_}")
# 打印降维后的数据，即将原始数据X投影到主成分上得到的新的数据集
print(f"降维后的数据:\n{newX}")
