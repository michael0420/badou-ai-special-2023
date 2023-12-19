from sklearn.decomposition import PCA

#示例数据
data = [[1,2,3],[4,5,6],[7,8,9]]

# 创建PCA对象，指定主成分的数量
pca = PCA(n_components=2)

#使用示例数据来拟合PCA模型
pca.fit(data)

#获取转化后的数据
transformed_data = pca.transform(data)

print(transformed_data)

