from signals_dataset import SignalsDataset

# torch related libraries
from cgi import test
import torch
from torch.nn import Linear
from torch_geometric.nn import MessagePassing

# functionalities
from split_data import split_data
from GSO import correlation_matrix
from MSELoss import movieMSELoss
from torch.utils.data import DataLoader


# other needed libraries
import pandas as pd
import numpy as np
import pickle
import math
import os
import matplotlib.pyplot as plt
import json

TARGET_MOVIES = [1] #[i for i in range(10)]#[i for i in range(9724)]
KNN = 3 #40
TRAIN_SPLIT = 0.5
N_EPOCHS = 20

# Preprocess data
df_ratings = pd.read_csv("../data/raw/small-test/ratings_2.csv")

movie_mapping = {idx: i for i, idx in enumerate(set(df_ratings.movieId.unique()))}
df_ratings["movieId"] = [movie_mapping[idx] for idx in df_ratings["movieId"]]
user_mapping = {idx: i for i, idx in enumerate(set(df_ratings.userId.unique()))}
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

GSO_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "precomputed_GSO/pearson_correlation.pkl")
train_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/train")
test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/test")

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

if not os.path.exists(GSO_filepath):
    correlation_matrix(X, idxTrain, KNN, N_movies, GSO_filepath)

file_to_read = open(GSO_filepath, 'rb')
data = pickle.load(file_to_read)
file_to_read.close()

edge_index, edge_weights = data['edge_index'], data['edge_weights']

split_data(X, idxTrain, idxTest, TARGET_MOVIES, train_dir, test_dir)

train_dataset = SignalsDataset(root_dir=train_dir)
test_dataset = SignalsDataset(root_dir=test_dir)

train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)


with open(f"{train_dir}/info.json", 'r') as outfile:
    info = json.load(outfile)
    samples_train = info["size"]

with open(f"{test_dir}/info.json", 'r') as outfile:
    info = json.load(outfile)
    samples_test = info["size"]


print("Number of train signals: " + str(samples_train))
print("Number of testing signals: " + str(samples_test))


class MyConv(MessagePassing):
    def __init__(self, in_features=1, out_features=64, K=5):
        super().__init__(aggr='add')
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_features, self.K, self.in_features))
        self.bias = torch.nn.parameter.Parameter(torch.Tensor(self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index, edge_weight):
        # conv dimensions == B * F_in * K * N
        conv = x.permute(1, 0).reshape([-1, self.in_features, 1, N_movies])
        print(x)
        for k in range(1, self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            
            print("new X")
            print(x)
            print('*********************************')
            x_aux = x.permute(1, 0).reshape([-1, self.in_features, 1, N_movies])
            conv = torch.cat((conv, x_aux), dim=2)

        # Actually multiply by the parameters
        # Reshape conv must be KG x F ===> order to B x N_movies x K x in_features and  reshape to B x N x (K*in_features)
        reshaped_conv = conv.permute(0, 3, 2, 1).reshape([-1, N_movies, self.K*self.in_features])

        # h convert KG x F
        h = self.weight.reshape([self.out_features, self.K*self.in_features]).permute(1, 0)

        y = torch.matmul(reshaped_conv, h)
        if self.bias is not None:
            y = y + self.bias
        return y

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MyConv()

    def forward(self, x, edge_index, edge_weights):
        x = self.conv1(x, edge_index, edge_weights).relu()
        return x


class Decoder(torch.nn.Module):
    def __init__(self, out_features=64, dim_readout=1):
        super().__init__()
        self.lin1 = Linear(out_features, dim_readout)

    def forward(self, z):
        z = self.lin1(z)
        z = z.reshape(-1, N_movies).permute(1,0)
        return z


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        y = self.encoder(x_dict, edge_index_dict, edge_label_index)
        z_dict = self.decoder(y)
        return z_dict


model = Model()

with torch.no_grad():
    model.encoder(next(iter(train_dataloader))[0].float().t(), edge_index, edge_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)

print("edge_index: " + str(edge_index.shape))
print("edge_weights: " + str(edge_weights.shape))


def train_step(x, y):
    model.train()
    optimizer.zero_grad()
    pred = model(x, edge_index, edge_weights)
    target = y
    loss = movieMSELoss(pred, target, TARGET_MOVIES)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def eval(x, y):
    model.eval()
    pred = model(x, edge_index, edge_weights)
    pred = pred.clamp(min=0, max=5)
    rmse = movieMSELoss(pred, y, TARGET_MOVIES)
    return float(rmse)


best_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/gps_pearson_best.pth")
last_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/gps_pearson_last.pth")
last_train_accuracy = 0
best_train_accuracy = float('inf')
last_test_accuracy = 0
best_test_accuracy = float('inf')

train_history = []
test_history = []

for epoch in range(1, N_EPOCHS):
    train_rmse = 0
    total_train_steps = 0
    for _, data in enumerate(train_dataloader):
        xTrain, yTrain = data
        xTrain = xTrain.float().t()
        print(xTrain)
        input()
        yTrain = yTrain.float().t()
        print(xTest)
        input()
        train_rmse += train_step(xTrain, yTrain)
        total_train_steps += 1
    mean_train = float(train_rmse / total_train_steps)
    train_history.append(mean_train)

    test_rmse = 0
    total_test_steps = 0
    for _, data in enumerate(test_dataloader):
        xTest, yTest = data
        xTest = xTest.float().t()
        yTest = yTest.float().t()
        test_rmse += eval(xTest, yTest)
        total_test_steps += 1
    mean_test = float(test_rmse / total_test_steps)
    test_history.append(mean_test)
    scheduler.step(mean_test)

    if mean_test < best_test_accuracy:
        best_train_accuracy = mean_train
        best_test_accuracy = mean_test
        torch.save(model, best_model_path)

    if epoch == N_EPOCHS - 1:
        last_train_accuracy = mean_train
        last_test_accuracy = mean_test
        torch.save(model, last_model_path)

    lr = optimizer.state_dict()['param_groups'][0]['lr']
    #if epoch % 10 == 0:
    print(f'Epoch: {epoch:03d}, Train_rmse: {mean_train:.4f}, Test_rmse: {mean_test:.4f}, LR: {lr:.10f}')

print("Finished trainning...Evaluating model")
print(f"Best model has train RMSE: {best_train_accuracy}")
print(f"Best model has test RMSE: {best_test_accuracy}")
print(f"Last model has train RMSE: {last_train_accuracy}")
print(f"Last model has test RMSE: {last_test_accuracy}")

x_axis = list(range(1, N_EPOCHS, 1))

plt.plot(x_axis, train_history, label="Train RMSE")
plt.plot(x_axis, test_history, label="Test RMSE")
plt.legend()
plt.show()
