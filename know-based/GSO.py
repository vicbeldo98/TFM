import time
import numpy as np
import torch
import pickle
import scipy
import pandas as pd

hardlycorrelated = 0.2


# Construct adyacency matrix
def correlation_matrix(X, idxTrain, knn, N_movies, filepath):
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
    invSqrtDiagonal = 1 / sqrtDiagonal
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
    W = W / np.max(np.abs(E))

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
    edge_weights = torch.tensor(weights).float()

    data = {
        'edge_index': edge_index,
        'edge_weights': edge_weights
    }

    file_to_write = open(filepath, 'wb')
    pickle.dump(data, file_to_write)
    file_to_write.close()


# Pearson correlation
def pearson_correlation(X, idxTrain, KNN, N, filepath):
    start = time.time()
    XTrain = X[idxTrain, :]
    df = pd.DataFrame(XTrain)
    W = df.corr(method='pearson')
    W[np.isnan(W)] = 0
    W = np.matrix(W)

    '''
    Queremos encontrar la similitud entre peliculas a partir de los ratings de los usuarios:

    +1 => ambas se mueven en la misma dirección (si te gusta una peli te suele gustar la otra) => podemos decir que son similares
     0 => no se encuentra correlación entre las variables
    -1 => se mueven en direcciones opuestas (si te gusta esta, no te suele gustar la otra y viceversa). (ESTO ES CORRELACIÓN PERO QUIZÁ NO NOS INTERESA. NO DENOTA SIMILITUD)

    '''
    W[W < 0] = 0
    W[np.abs(W) < hardlycorrelated] = 0.
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
    edge_weights = torch.tensor(weights).float()

    data = {
        'edge_index': edge_index,
        'edge_weights': edge_weights
    }

    file_to_write = open(filepath, 'wb')
    pickle.dump(data, file_to_write)
    file_to_write.close()


#   TODO: Finish this implementations and make comparation
#   TODO: give option to sum identity matrix
#   TODO: normalize entry values between 0 and 1

def adjacency_matrix(X, idxTrain, knn, N_movies, filepath):
    XTrain = X[idxTrain, :]
    df = pd.DataFrame(XTrain)
    W = df.corr(method='pearson')
    W[np.isnan(W)] = 0
    W = np.matrix(W)

    '''
    Queremos encontrar la similitud entre peliculas a partir de los ratings de los usuarios:

    +1 => ambas se mueven en la misma dirección (si te gusta una peli te suele gustar la otra)
     0 => no se encuentra correlación entre las variables
    -1 => se mueven en direcciones opuestas (si te gusta esta, no te suele gustar la otra y viceversa). (ESTO ES CORRELACIÓN PERO QUIZÁ NO NOS INTERESA. NO DENOTA SIMILITUD)

    '''
    W[W < 0] = 0
    print(W)
    input()
    print(W.shape[0] - np.count_nonzero(W.sum(axis=1)))
    input()
    W[W < hardlycorrelated] = 0.

    W[W != 0] = 1

    print(W)
    input()
    print(W.sum(axis=1))
    input()


def adjacency_normalized_matrix(X, idxTrain, knn, N_movies, filepath):
    pass


def laplacian_matrix(X, idxTrain, knn, N_movies, filepath):
    pass


def laplacian_normalized_matrix(X, idxTrain, knn, N_movies, filepath):
    pass
