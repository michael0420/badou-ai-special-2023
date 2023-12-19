#!/usr/bin/env python
# encoding=utf-8
import pandas as pd


data = pd.read_csv("data/train_data.csv", sep='\\s*,\\s*', engine="python")
x, y = data.X, data.Y
n = len(data)
sum_xy = sum(x * y)
sum_product = sum(x) * sum(y)
sum_x2 = sum(x ** 2)
square_sum_x = sum(x) ** 2
mean_x = sum(x) / n
mean_y = sum(y) / n

k = (n * sum_xy - sum_product) / (n * sum_x2 - square_sum_x)
b = mean_y - k * mean_x

print(k, b)
