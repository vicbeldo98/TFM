import os
import numpy as np

train_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/train/signals")

for i in range(1, 449):
    with open(f"{train_dir}/{i}.npy", 'rb') as f:
        if np.load(f).shape[0] != 50:
            print(i)
