import numpy as np

X = np.array([[1, 2, 3, 4, 0], [0, 0, 2, 5, 8], [0, 0, 0, 0, 0], [0, 1, 2, 3, 4]])
print((X[:, 4] != 0).nonzero()[0])
