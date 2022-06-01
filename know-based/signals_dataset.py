import json
from torch.utils.data import Dataset
import math
import numpy as np


class SignalsDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.root_dir = root_dir

    def __len__(self):
        with open(f"{self.root_dir}/info.json", 'r') as outfile:
            info = json.load(outfile)

        return info["size"]

    def __getitem__(self, idx):

        check_file = math.floor(idx / 50) + 1
        with open(f"{self.root_dir}/info.json", 'r') as outfile:
            info = json.load(outfile)

        signals_dir, labels_dir = info[f"{check_file}"][0], info[f"{check_file}"][1]

        with open(signals_dir, 'rb') as f:
            xTrain = np.load(f)

        with open(labels_dir, 'rb') as f:
            xTest = np.load(f)

        row = idx % 50

        return (xTrain[row, :], xTest[row, :])


'''
train_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/train")
dataset = SignalsDataset(root_dir=train_dir)
train_dataloader = DataLoader(dataset, batch_size=512, shuffle=False)


print(next(iter(train_dataloader)))
xTrain, xTest = next(iter(train_dataloader))
print(f"Feature batch shape: {xTrain.size()}")
print(f"Feature batch shape: {xTest.size()}")
'''
