import re
from signals_dataset import SignalsDataset

# torch related libraries
import torch
from torch.nn import Linear
from torch_geometric.nn import MessagePassing

# functionalities
from split_data import split_data
from GSO import pearson_correlation, adjacency_matrix, adjacency_normalized_matrix, laplacian_matrix, laplacian_normalized_matrix
from MSELoss import movieMSELoss   # movieRMSELoss
from torch.utils.data import DataLoader


# other needed libraries
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import json
import argparse
import math

TARGET_MOVIES = [257]
KNN = 500
TRAIN_SPLIT = 0.85
N_EPOCHS = 40
VERBOSE = False

classes = [1.0, 2.0, 3.0, 4.0, 5.0]

def to_categorical(batch, idx):
    target = batch[:, idx]
    new_target = []
    for i in target:
        aux = np.zeros(len(classes))
        pos = classes.index(i)
        aux[pos] = 1
        new_target.append(aux)
    res = torch.tensor(np.array(new_target))
    return res


def calc_accuracy(pred, target):
    total = 0
    correct = 0
    for i in range(pred.shape[0]):
        pos = np.where(pred[i] == max(pred[i]))[0][0]
        pred_class = classes[pos]
        pos2 = np.where(target[i] != 0)[0][0]
        target_class = classes[pos2]
        if pred_class == target_class:
            correct += 1
        total += 1
    return float(correct / total)

parser = argparse.ArgumentParser()
parser.add_argument('-gso', metavar="GSO matrix")
parsed_args = parser.parse_args()
args = vars(parsed_args)

# Preprocess data
df_ratings = pd.read_csv("../data/raw/ml-100K/ratings.csv")

movie_mapping = {idx: i for i, idx in enumerate(set(df_ratings.movieId.unique()))}
df_ratings["movieId"] = [movie_mapping[idx] for idx in df_ratings["movieId"]]
user_mapping = {idx: i for i, idx in enumerate(set(df_ratings.userId.unique()))}
df_ratings["userId"] = [user_mapping[idx] for idx in df_ratings["userId"]]

TARGET_MOVIES = [movie_mapping[i] for i in TARGET_MOVIES]


# Split data into train and test
nTotal = len(user_mapping.keys())
if VERBOSE:
    print("Number of users: " + str(nTotal))
    print("Number of movies: " + str(len(movie_mapping.keys())))

permutation = np.random.RandomState(seed=42).permutation(np.arange(nTotal))
nTrain = int(np.ceil(TRAIN_SPLIT * nTotal))
idxTrain = permutation[0:nTrain]
nTest = nTotal - nTrain
idxTest = permutation[nTrain:nTotal]

print("Number of total signals to train: " + str(nTrain))
print("Number of total signals to test: " + str(nTest))

N_users = len(user_mapping.keys())
N_movies = len(movie_mapping.keys())

# Construct matrix users x movies with ratings as entries
X = np.zeros((N_users, N_movies))
for idx, row in df_ratings.iterrows():
    X[int(row["userId"]), int(row["movieId"])] = row["rating"] #- 1.0) / 4.0

GSO_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "precomputed_GSO/gso.pkl")
train_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/train")
test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/test")

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

if not os.path.exists(GSO_filepath):
    if args['gso'] == "a":
        adjacency_matrix(X, idxTrain, KNN, GSO_filepath)
    elif args['gso'] == "l":
        laplacian_matrix(X, idxTrain, KNN, GSO_filepath)
    elif args['gso'] == "na":
        adjacency_normalized_matrix(X, idxTrain, KNN, GSO_filepath)
    elif args['gso'] == "nl":
        laplacian_normalized_matrix(X, idxTrain, KNN, GSO_filepath)
    else:
        pearson_correlation(X, idxTrain, KNN, GSO_filepath)

file_to_read = open(GSO_filepath, 'rb')
data = pickle.load(file_to_read)
file_to_read.close()

edge_index, edge_weights = data['edge_index'], data['edge_weights']

split_data(X, idxTrain, idxTest, TARGET_MOVIES, train_dir, test_dir)

train_dataset = SignalsDataset(root_dir=train_dir)
test_dataset = SignalsDataset(root_dir=test_dir)

train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)

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
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_features, self.K, self.in_features), requires_grad=True)
        self.bias = torch.nn.parameter.Parameter(torch.Tensor(self.out_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index, edge_weight):
        # conv dimensions == B * F_in * K * N
        if VERBOSE:
            print("Sample:")
            print(x.shape)
            print(x)
        conv = x.permute(1, 0).reshape([-1, self.in_features, 1, N_movies])

        for k in range(1, self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            if VERBOSE:
                print(f"Propagation {k}:")
                print(x)
            x_aux = x.permute(1, 0).reshape([-1, self.in_features, 1, N_movies])
            conv = torch.cat((conv, x_aux), dim=2)

        # Actually multiply by the parameters
        # Reshape conv must be KG x F ===> order to B x N_movies x K x in_features and  reshape to B x N x (K*in_features)
        reshaped_conv = conv.permute(0, 3, 2, 1).reshape([-1, N_movies, self.K * self.in_features])
        if VERBOSE:
            print("Matrix of accumulated embeddings:")
            print(reshaped_conv.shape)
            print(reshaped_conv)

        # h convert KG x F
        h = self.weight.reshape([self.out_features, self.K * self.in_features]).permute(1, 0)

        y = torch.matmul(reshaped_conv, h)

        if self.bias is not None:
            y = y + self.bias
        if VERBOSE:
            print("Final result of forward")
            print(y.shape)
            print(y)
        return y

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MyConv()
        self.conv1_bn=torch.nn.BatchNorm1d(N_movies)


    def forward(self, x, edge_index, edge_weights):
        print("ENCODER")
        x = self.conv1(x, edge_index, edge_weights)
        x = self.conv1_bn(x).relu()
        print("ENCODER END")
        return x


class Decoder(torch.nn.Module):
    def __init__(self, out_features=64, dim_readout=1):
        super().__init__()
        self.lin1 = Linear(out_features, dim_readout)
        self.lin2 = Linear(N_movies, len(classes))


    def forward(self, x):
        x = self.lin1(x).relu()
        x = x.reshape(-1, N_movies)
        x = self.lin2(x)
        output_layer = torch.nn.Softmax(dim=1)
        x = output_layer(x)
        return x


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, edge_index_dict, edge_label_index):
        x = self.encoder(x, edge_index_dict, edge_label_index)
        x = self.decoder(x)
        # z_dict = z_dict.clone().detach().requires_grad_(True)
        if VERBOSE:
            print("After linear layer and reshape:")
            print(x.shape)
            print(x)
        return x


model = Model()

with torch.no_grad():
    model.encoder(next(iter(train_dataloader))[0].float().t(), edge_index, edge_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)

if VERBOSE:
    print("edge_index: " + str(edge_index.shape))
    print("edge_weights: " + str(edge_weights.shape))

def train_step(x, y):
    model.train()
    pred = model(x, edge_index, edge_weights)
    target = to_categorical(y, TARGET_MOVIES[0])
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return calc_accuracy(pred, target)


@torch.no_grad()
def eval(x, y):
    model.eval()
    pred = model(x, edge_index, edge_weights)
    target = to_categorical(y, TARGET_MOVIES[0])
    acc = calc_accuracy(pred, target)
    return float(acc)


best_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/gps_pearson_best.pth")
last_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/gps_pearson_last.pth")
last_train_accuracy = 0
best_train_accuracy = float('inf')
last_test_accuracy = 0
best_test_accuracy = float('inf')

train_history = []
test_history = []

for epoch in range(1, N_EPOCHS):
    train_loss = 0
    total_train_steps = 0
    for _, data in enumerate(train_dataloader):
        xTrain, yTrain = data
        xTrain = xTrain.float().t()
        yTrain = yTrain.float()
        train_loss += train_step(xTrain, yTrain)
        total_train_steps += 1
    mean_train = float(train_loss / total_train_steps)
    train_history.append(mean_train)
    test_loss = 0
    total_test_steps = 0
    for _, data in enumerate(test_dataloader):
        xTest, yTest = data
        xTest = xTest.float().t()
        yTest = yTest.float()
        test_loss += eval(xTest, yTest)
        total_test_steps += 1
    mean_test = float(test_loss / total_test_steps)
    test_history.append(mean_test)
    scheduler.step(mean_test)

    '''if mean_test < best_test_accuracy:
        best_train_accuracy = mean_train
        best_test_accuracy = mean_test
        torch.save(model, best_model_path)

    if epoch == N_EPOCHS - 1:
        last_train_accuracy = mean_train
        last_test_accuracy = mean_test
        torch.save(model, last_model_path)'''

    lr = optimizer.state_dict()['param_groups'][0]['lr']
    print(f'Epoch: {epoch:03d}, Train_rmse: {mean_train:.4f}, Test_rmse: {mean_test:.4f}, LR: {lr:.10f}')

print("Finished trainning...Evaluating model")
print(f"Best model has train RMSE: {best_train_accuracy}")
print(f"Best model has test RMSE: {best_test_accuracy}")
print(f"Last model has train RMSE: {last_train_accuracy}")
print(f"Last model has test RMSE: {last_test_accuracy}")

x_axis = list(range(1, N_EPOCHS, 1))

plt.plot(x_axis, train_history, label="Train RMSE")
plt.plot(x_axis, test_history, label="Test RMSE")
plt.axis([1, N_EPOCHS, 0, 10])
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.title("Evolution of RMSE in training for Men in Black (1997)")
plt.legend()
plt.show()
