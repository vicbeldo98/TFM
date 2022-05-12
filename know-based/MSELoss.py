import torch

def movieMSELoss(yHat, y, idxMovie):
    mse = torch.nn.MSELoss()
    return mse(yHat[idxMovie, :], y[idxMovie, :])