import numpy as np
import torch
import pickle
import pandas as pd
import math


#   TODO: Finish this implementations and make comparation
#   TODO: give option to sum identity matrix
#   TODO: normalize entry values between 0 and 1
#   TODO: use knn in the construction of W

'''
    Queremos encontrar la similitud entre peliculas a partir de los ratings de los usuarios:

    +1 => ambas se mueven en la misma dirección (si te gusta una peli te suele gustar la otra)
        0 => no se encuentra correlación entre las variables
    -1 => se mueven en direcciones opuestas (si te gusta esta, no te suele gustar la otra y viceversa).
    (ESTO ES CORRELACIÓN PERO QUIZÁ NO NOS INTERESA. NO DENOTA SIMILITUD)

'''


hardlycorrelated = 0.2


def compute_pearson(XTrain):
    df = pd.DataFrame(XTrain)
    W = df.corr(method='pearson')
    W[np.isnan(W)] = 0
    W = np.matrix(W)
    return W


def to_COO(W):
    src = []
    dst = []
    weights = []
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            src.append(i)
            dst.append(j)
            weights.append(W[i, j])

    edge_index = torch.LongTensor(np.array([src, dst]))
    edge_weights = torch.tensor(weights).float()

    data = {
        'edge_index': edge_index,
        'edge_weights': edge_weights
    }

    return data


def pearson_correlation(X, idxTrain, knn, filepath):
    XTrain = X[idxTrain, :]
    W = compute_pearson(XTrain)

    # Filtrar valores negativos
    W[W < 0] = 0

    # Filtrar valores con correlación muy pequeña
    W[W < hardlycorrelated] = 0

    print("Pearson correlation")
    print(W)

    data = to_COO(W)

    file_to_write = open(filepath, 'wb')
    pickle.dump(data, file_to_write)
    file_to_write.close()


def adjacency_matrix(X, idxTrain, knn, filepath):
    XTrain = X[idxTrain, :]
    W = compute_pearson(XTrain)

    # Filtrar valores negativos
    W[W < 0] = 0

    # Filtrar valores con correlación muy pequeña
    W[W < hardlycorrelated] = 0

    # Todos los valores que no sean 0 son 1 (OBJETIVO: matriz de adyacencia)
    W[W != 0] = 1

    print("Matriz de adyacencia")
    print(W)

    data = to_COO(W)

    file_to_write = open(filepath, 'wb')
    pickle.dump(data, file_to_write)
    file_to_write.close()


def laplacian_matrix(X, idxTrain, knn, filepath):
    XTrain = X[idxTrain, :]
    A = compute_pearson(XTrain)

    # Filtrar valores negativos
    A[A < 0] = 0

    # Filtrar valores con correlación muy pequeña
    A[A < hardlycorrelated] = 0

    # Todos los valores que no sean 0 son 1 (OBJETIVO: matriz de adyacencia)
    A[A != 0] = 1

    diag_matrix = np.zeros(A.shape)

    for row in range(A.shape[0]):
        number = np.sum(A[row])
        diag_matrix[row][row] = number

    laplacian_matrix = diag_matrix - A
    print("Laplacian matrix")
    print(laplacian_matrix)

    data = to_COO(laplacian_matrix)

    file_to_write = open(filepath, 'wb')
    pickle.dump(data, file_to_write)
    file_to_write.close()


def adjacency_normalized_matrix(X, idxTrain, knn, filepath):
    XTrain = X[idxTrain, :]
    A = compute_pearson(XTrain)

    # Filtrar valores negativos
    A[A < 0] = 0

    # Filtrar valores con correlación muy pequeña
    A[A < hardlycorrelated] = 0

    # Todos los valores que no sean 0 son 1 (OBJETIVO: matriz de adyacencia)
    A[A != 0] = 1

    diag_matrix = np.zeros(A.shape)

    for row in range(A.shape[0]):
        number = np.sum(A[row])
        power_inverse = 1 / (math.sqrt(number))
        diag_matrix[row][row] = power_inverse

    adjacency_normalized_matrix = diag_matrix * A * diag_matrix
    print("Adjacency normalized matrix")
    print(adjacency_normalized_matrix)

    data = to_COO(adjacency_normalized_matrix)

    file_to_write = open(filepath, 'wb')
    pickle.dump(data, file_to_write)
    file_to_write.close()


def laplacian_normalized_matrix(X, idxTrain, knn, filepath):
    XTrain = X[idxTrain, :]
    A = compute_pearson(XTrain)

    # Filtrar valores negativos
    A[A < 0] = 0

    # Filtrar valores con correlación muy pequeña
    A[A < hardlycorrelated] = 0

    # Todos los valores que no sean 0 son 1 (OBJETIVO: matriz de adyacencia)
    A[A != 0] = 1

    diag_matrix = np.zeros(A.shape)

    for row in range(A.shape[0]):
        number = np.sum(A[row])
        diag_matrix[row][row] = number

    laplacian_matrix = diag_matrix - A

    diag_matrix = np.zeros(A.shape)

    for row in range(A.shape[0]):
        number = np.sum(A[row])
        power_inverse = 1 / (math.sqrt(number))
        diag_matrix[row][row] = power_inverse

    laplacian_normalized_matrix = diag_matrix * laplacian_matrix * diag_matrix
    print("Laplacian normalized matrix")
    print(laplacian_normalized_matrix)
    data = to_COO(laplacian_normalized_matrix)

    file_to_write = open(filepath, 'wb')
    pickle.dump(data, file_to_write)
    file_to_write.close()
