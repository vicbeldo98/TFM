import torch
import math


def movieMSELoss(yHat, y):
    print(yHat)
    mse = torch.nn.MSELoss()
    return mse(yHat, y)
