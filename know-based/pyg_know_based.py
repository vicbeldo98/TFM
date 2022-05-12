import torch
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from split_data import split_data

from GSO import correlation_matrix
from MSELoss import movieMSELoss

import pandas as pd
import numpy as np

TARGET_MOVIES = 257
KNN = 40
TRAIN_SPLIT = 0.85

# Preprocess data
df_ratings = pd.read_csv("../data/raw/ml-latest-small/ratings.csv")

movie_mapping = {idx: i for i, idx in enumerate(df_ratings.movieId.unique())}
df_ratings["movieId"] = [movie_mapping[idx] for idx in df_ratings["movieId"]]
user_mapping = {idx: i for i, idx in enumerate(df_ratings.userId.unique())}
df_ratings["userId"] = [user_mapping[idx] for idx in df_ratings["userId"]]

# Split data into train and test
nTotal = len(user_mapping.keys())
print("Number of users: " + str(nTotal))
print("Number of movies: " + str(len(movie_mapping.keys())))
print("Number of ratings: " + str(df_ratings.shape[0]))

permutation = np.random.RandomState(seed=42).permutation(np.arange(nTotal))
nTrain = int(np.ceil(TRAIN_SPLIT * nTotal))
idxTrain = permutation[0:nTrain]
nTest = nTotal - nTrain
idxTest = permutation[nTrain:nTotal]
print("Number of signals to train: " + str(nTrain))
print("Number of signals to test: " + str(nTest))

N_users = len(user_mapping.keys())
N_movies = len(movie_mapping.keys())

# Construct matrix users x movies with ratings as entries
X = np.zeros((N_users, N_movies))
for idx, row in df_ratings.iterrows():
    X[int(row["userId"]), int(row["movieId"])] = row["rating"]

edge_index, edge_weights = correlation_matrix(X, idxTrain, KNN, N_movies, N_users)
xTrain, yTrain, xTest, yTest = split_data(X, idxTrain, idxTest, TARGET_MOVIES)
nTrain = xTrain.shape[0]
nTest = xTest.shape[0]
print("Number of training samples: " + str(nTrain))
print("Number of testing samples: " + str(nTest))
print("xTrain: " + str(xTrain.shape))
print("yTrain: " + str(yTrain.shape))
print("xTest: " + str(xTest.shape))
print("yTest: " + str(yTest.shape))


class MyConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
        self.weight = torch.nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(1)

    def forward(self, x, edge_index, edge_weight):
        return x + self.weight * self.propagate(edge_index, x=x,
                                                edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MyConv()
        self.conv2 = MyConv()
        self.conv3 = MyConv()

    def forward(self, x, edge_index, edge_weights):
        x = self.conv1(x, edge_index, edge_weights).relu()
        #   x = self.conv2(x, edge_index, edge_weights)
        #   x = self.conv3(x, edge_index, edge_weights)
        return x


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(2830, 2830)

    def forward(self, z):
        z = z.t()
        z = self.lin1(z)
        return z.view(-1, 1)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        #   self.decoder = Decoder()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict, edge_label_index)
        #   z_dict = self.decoder(z_dict)
        return z_dict


model = Model()

with torch.no_grad():
    model.encoder(xTrain, edge_index, edge_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       patience=20)

print("data: " + str(xTrain.shape))
print("edge_index: " + str(edge_index.shape))
print("edge_weights: " + str(edge_weights.shape))


def train():
    model.train()
    optimizer.zero_grad()
    pred = model(xTrain, edge_index, edge_weights)
    target = yTrain
    loss = movieMSELoss(pred, target, TARGET_MOVIES)
    loss.backward()
    scheduler.step(loss)
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(xTest, edge_index, edge_weights)
    pred = pred.clamp(min=0, max=5)
    target = yTest
    rmse = movieMSELoss(pred, target, TARGET_MOVIES)
    return float(rmse)


for epoch in range(1, 101):
    loss = train()
    test_rmse = test()
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test: {test_rmse:.4f}, LR: {lr:.10f}')
