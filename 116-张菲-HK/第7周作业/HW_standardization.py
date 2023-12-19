import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def max_min(x):
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]


def max_mean(x):
    return [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]


def z_score(x):
    x_mean = np.mean(x)
    s2 = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)
    return [(i - x_mean) / s2 for i in x]

np.random.seed(123)
data = np.random.randn(1000)

n1 = max_min(data)
n2 = max_mean(data)
n3 = z_score(data)

sns.kdeplot(data, label="original", color='green', shade=True)
sns.kdeplot(n1, label='max min', color="LightPink", shade=True)
sns.kdeplot(n2, label='max mean', color="DarkMagenta", shade=True)
sns.kdeplot(n3, label='z score', color="SkyBlue", shade=True)

plt.title('Density Estimation Plot')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)

plt.legend()
plt.show()
