import numpy as np

if __name__ == '__main__':
    data_to_normalize = np.array([18.0, 28.0, 13.0, 48.0, 13.0, 61.0, 18.0, 38.0])
    avg = np.average(data_to_normalize)
    std = np.std(data_to_normalize)
    normalized_data = []
    for x in data_to_normalize:
        x = (x - avg) / std
        normalized_data.append(x)
    print(np.array(normalized_data))
