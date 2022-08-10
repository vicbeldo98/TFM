import torch
import math


def movieMSELoss(yHat, y):
    mse = torch.nn.MSELoss()
    return mse(yHat, y)
