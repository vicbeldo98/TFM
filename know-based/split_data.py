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