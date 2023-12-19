import numpy as np


def normalization_min(x):
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]


def normalization_mean(x):
    return [(float(i) - np.mean(x)) / float(max(x) - min(x)) for i in x]


def standard(x):
    """x=(x-μ)/σ2"""
    x_mean = np.mean(x)
    s2 = sum([(i - x_mean) ** 2 for i in x]) / len(x)
    # x_sqrt = np.sqrt(s2)
    return [(i - x_mean) / s2 for i in x]


test = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10,
        10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14,
        14, 15, 15, 30,
        ]
print('normalization_min', normalization_min(test))
print('normalization_mean', normalization_mean(test))
print('standard', standard(test))
