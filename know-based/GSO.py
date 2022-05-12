import time
import numpy as np
import torch
import scipy

# Construct adyacency matrix
def correlation_matrix(X, idxTrain, knn, N_movies, N_users):
    start = time.time()
    zeroTolerance = 1e-9
    # Construct a matrix movies x users with train data
    XTrain = np.transpose(X[idxTrain, :])

    # Calculating correlation matrix
    binaryTemplate = (XTrain > 0).astype(XTrain.dtype)
    sumMatrix = XTrain.dot(binaryTemplate.T)
    countMatrix = binaryTemplate.dot(binaryTemplate.T)
    countMatrix[countMatrix == 0] = 1
    avgMatrix = sumMatrix / countMatrix
    sqSumMatrix = (XTrain ** 2).dot(binaryTemplate.T)
    correlationMatrix = sqSumMatrix / countMatrix - avgMatrix ** 2

    # Normalizing by diagonal weights
    sqrtDiagonal = np.sqrt(np.diag(correlationMatrix))
    nonzeroSqrtDiagonalIndex = (sqrtDiagonal > zeroTolerance).astype(sqrtDiagonal.dtype)
    sqrtDiagonal[sqrtDiagonal < zeroTolerance] = 1.
    invSqrtDiagonal = 1/sqrtDiagonal
    invSqrtDiagonal = invSqrtDiagonal * nonzeroSqrtDiagonalIndex
    normalizationMatrix = np.diag(invSqrtDiagonal)

    # Zero-ing the diagonal
    normalizedMatrix = normalizationMatrix.dot(correlationMatrix.dot(normalizationMatrix)) - np.eye(correlationMatrix.shape[0])

    # Keeping only edges with weights above the zero tolerance
    normalizedMatrix[np.abs(normalizedMatrix) < zeroTolerance] = 0.
    W = normalizedMatrix

    # Sparsifying the graph
    WSorted = np.sort(W, axis=1)
    threshold = WSorted[:, -knn].squeeze()
    thresholdMatrix = (np.tile(threshold, (N_movies, 1))).transpose()
    W[W < thresholdMatrix] = 0

    # Normalizing by eigenvalue with largest magnitude
    E, V = scipy.sparse.linalg.eigs(W)
    W = W/np.max(np.abs(E))
    end = time.time()
    time_spent = end - start
    print("Time spent computing the correlation matrix: " + str(time_spent) + "s")

    src = []
    dst = []
    weights = []
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i][j] > 0.001:
                src.append(i)
                dst.append(j)
                weights.append(W[i][j])

    edge_index = torch.LongTensor(np.array([src, dst]))
    return edge_index, torch.tensor(weights).float()