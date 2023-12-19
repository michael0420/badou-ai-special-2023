import numpy as np
import sklearn.decomposition as dp

# 随机自定义输入数据，格式是(10,4)
input = np.random.randint(0, 100, size=(10, 4))
# 调用PCA接口
down = dp.PCA(n_components=2)
# 利用接口降维度
output = down.fit_transform(input)

print("输入数组维度：", input.shape)
print("调用PCA接口降维后维度", output.shape)


