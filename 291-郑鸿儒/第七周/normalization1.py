#!/usr/bin/env python
# encoding=utf-8
import numpy as np


def normalization_min(x):
    return [(float(i) - min(x)) / (max(x) - min(x)) for i in x]


def normalization_mean(x):
    return [(float(i) - np.mean(x)) / (max(x) - np.mean(x)) for i in x]


def normalization_z_score(x):
    s2 = sum([(i - np.mean(x)) ** 2 for i in x]) / len(x)
    return [(float(i) - np.mean(x)) / s2 for i in x]

