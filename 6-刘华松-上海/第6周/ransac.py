import numpy as np
import random

def ransac(data, model, n, k, t, d):
    """
    RANSAC算法实现
    data: 输入数据
    model: 数据模型，可以是线性模型、多项式模型等
    n: 适用于模型的最小数据点数量
    k: 迭代次数
    t: 阈值，用于确定一个数据点适合模型的条件
    d: 用于判断模型的好坏，即适合模型数据点的数量
    """
    best_model = None
    best_inliers = None
    
    for i in range(k):
        # 1. 从数据集中随机选择n个数据点
        random_indices = random.sample(range(len(data)), n)
        sampled_data = [data[i] for i in random_indices]
        
        # 2. 用随机选择的数据点估计模型参数
        model.fit(sampled_data)
        
        # 3. 计算所有数据点到估计模型的距离，并选择距离小于阈值t的数据点作为inliers
        inliers = []
        for j, point in enumerate(data):
            if model.distance(point) < t:
                inliers.append(j)
        
        # 4. 如果inliers数量大于阈值d，重新估计模型参数，更新最优模型
        if len(inliers) >= d:
            inlier_data = [data[j] for j in inliers]
            updated_model = model.fit(inlier_data)
            
            if len(inliers) > len(best_inliers):
                best_model = updated_model
                best_inliers = inliers
    
    return best_model, best_inliers