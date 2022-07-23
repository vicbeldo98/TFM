import numpy as np
from GSO import pearson_correlation, adjacency_matrix, adjacency_normalized_matrix, laplacian_matrix, laplacian_normalized_matrix


W = np.array([[5, 0, 3, 0], [2, 1, 5, 3], [3, 1, 5, 3], [3, 5, 1, 2]])

pearson_correlation(W, [0, 1, 2, 3], 2, "lala.npy")

adjacency_matrix(W, [0, 1, 2, 3], 2, "lala.npy")

adjacency_normalized_matrix(W, [0, 1, 2, 3], 2, "lala.npy")

laplacian_matrix(W, [0, 1, 2, 3], 2, "lala.npy")

laplacian_normalized_matrix(W, [0, 1, 2, 3], 2, "lala.npy")
