import numpy as  np

if __name__ == '__main__':
    data_to_normalize = np.array([18.0, 28.0, 13.0, 48.0, 13.0, 61.0, 18.0, 38.0])
    normalized_data = []
    for x in data_to_normalize:
        y = 1.0 / (1 + np.exp(-float(x)))
        normalized_data.append(y)
    print(np.array(normalized_data))

