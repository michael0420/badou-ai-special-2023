#!/usr/bin/env python
# encoding=utf-8
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
import matplotlib.pyplot as plt


X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
z = linkage(X, 'ward')
f = fcluster(z, 4, criterion='distance')
print(z)
print(f)
plt.figure(figsize=(6, 4))
dendrogram(z)
plt.show()
