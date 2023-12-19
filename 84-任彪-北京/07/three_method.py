from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

if __name__ == '__main__':
    data_to_normalize = np.array([18.0, 28.0, 13.0, 48.0, 13.0, 61.0,18.0, 38.0])

    # 1、 minmax 归一化
    # 创建MinMaxScaler对象
    # scaler = MinMaxScaler()
    # # 用fit_transform函数进行归一化
    # normalized_data = scaler.fit_transform(data_to_normalize.reshape(-1, 1))

    # 2、z-score
    normalized_data = preprocessing.scale(data_to_normalize)  # data是多维数据



    print("原始数据:\n", data_to_normalize)
    print("归一化后的数据:\n", normalized_data)