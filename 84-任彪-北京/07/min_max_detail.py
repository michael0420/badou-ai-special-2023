import numpy as np

if __name__ == '__main__':

    data_to_normalize = np.array([18.0, 28.0, 13.0, 48.0, 13.0, 61.0,18.0, 38.0])
    normalized_data = []
    for x in data_to_normalize:
        x = float(x - np.min(data_to_normalize)) / (np.max(data_to_normalize) - np.min(data_to_normalize))
        normalized_data.append(x)
    print(np.array(normalized_data))
