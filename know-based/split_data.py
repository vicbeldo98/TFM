import numpy as np
import torch

def split_data(X, idxTrain, idxTest, idxMovie):
    xTrain = X[idxTrain, :]
    idx = np.argwhere(xTrain[:, idxMovie] > 0).squeeze()
    xTrain = xTrain[idx, :]
    yTrain = np.zeros(xTrain.shape)
    yTrain[:, idxMovie] = xTrain[:, idxMovie]
    xTrain[:, idxMovie] = 0
    xTrain = torch.tensor(xTrain).float().t()
    yTrain = torch.tensor(yTrain).float().t()

    xTest = X[idxTest, :]
    idx = np.argwhere(xTest[:, idxMovie] > 0).squeeze()
    xTest = xTest[idx, :]
    yTest = np.zeros(xTest.shape)
    yTest[:, idxMovie] = xTest[:, idxMovie]
    xTest[:, idxMovie] = 0
    xTest = torch.tensor(xTest).float().t()
    yTest = torch.tensor(yTest).float().t()

    return xTrain, yTrain, xTest, yTest


def original_split_data(X, idxTrain, idxTest, idxMovie):  
    
    N = X.shape[1]
    
    xTrain = X[idxTrain,:]
    idx = np.argwhere(xTrain[:,idxMovie]>0).squeeze()
    xTrain = xTrain[idx,:]
    yTrain = np.zeros(xTrain.shape)
    yTrain[:,idxMovie] = xTrain[:,idxMovie]
    xTrain[:,idxMovie] = 0
    
    xTrain = torch.tensor(xTrain)
    xTrain = xTrain.reshape([-1,1,N])
    yTrain = torch.tensor(yTrain)
    yTrain = yTrain.reshape([-1,1,N])
    
    xTest = X[idxTest,:]
    idx = np.argwhere(xTest[:,idxMovie]>0).squeeze()
    xTest = xTest[idx,:]
    yTest = np.zeros(xTest.shape)
    yTest[:,idxMovie] = xTest[:,idxMovie]
    xTest[:,idxMovie] = 0
    
    xTest = torch.tensor(xTest)
    xTest = xTest.reshape([-1,1,N])
    yTest = torch.tensor(yTest)
    yTest = yTest.reshape([-1,1,N])
    
    return xTrain, yTrain, xTest, yTest