import numpy as np
import torch
import random

def split_data(X, idxTrain, idxTest, idxMovie):
    xTrain = X[idxTrain, :]
    '''idx = np.argwhere(xTrain[:, idxMovie] > 0).squeeze()
    xTrain = xTrain[idx, :]'''
    xTrain, yTrain = generate_signals(xTrain, idxMovie)
    '''yTrain = np.zeros(xTrain.shape)
    yTrain[:, idxMovie] = xTrain[:, idxMovie]
    xTrain[:, idxMovie] = 0'''
    xTrain = torch.tensor(xTrain).float().t()
    yTrain = torch.tensor(yTrain).float().t()

    xTest = X[idxTest, :]
    '''idx = np.argwhere(xTest[:, idxMovie] > 0).squeeze()
    xTest = xTest[idx, :]'''
    xTest, yTest = generate_signals(xTest, idxMovie)
    '''yTest = np.zeros(xTest.shape)
    yTest[:, idxMovie] = xTest[:, idxMovie]
    xTest[:, idxMovie] = 0'''
    xTest = torch.tensor(xTest).float().t()
    yTest = torch.tensor(yTest).float().t()

    return xTrain, yTrain, xTest, yTest


def generate_signals(X, idx):
    N_SAMPLES_PER_SIGNAL = 4

    x_signals = []
    x_labels = []

    for i in range(X.shape[0]):

        ref = X[i,:]

        target_ratings = np.take(ref, idx)

        index_target_ratings = np.where(target_ratings != 0)[0]

        if len(index_target_ratings) < N_SAMPLES_PER_SIGNAL:
            samples = len(index_target_ratings)
        else:
            samples = N_SAMPLES_PER_SIGNAL

        random_sampled_index = random.sample(set(index_target_ratings), samples)

        for idx2 in random_sampled_index:
            ref_signal = ref.copy()
            ref_signal[idx2] = 0
            x_signals.append(ref_signal)
            label = np.zeros(len(ref))
            label[idx2] = ref[idx2]
            x_labels.append(label)
    
    return np.array(x_signals), np.array(x_labels)
