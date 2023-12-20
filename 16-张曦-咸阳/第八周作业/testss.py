import numpy as np

arr = np.array([1, 2, 3])

arr_0 = np.array(arr, ndmin=0)
arr_1 = np.array(arr, ndmin=1)
arr_2 = np.array(arr, ndmin=2)

print("arr_0 shape:", arr_0.shape,arr_0)  # Output: arr_0 shape: (3,)
print("arr_1 shape:", arr_1.shape,arr_1)  # Output: arr_1 shape: (3,)
print("arr_2 shape:", arr_2.shape,arr_2)  # Output: arr_2 shape: (3, 1)