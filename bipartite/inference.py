import os.path as osp
import torch
import torch_geometric.transforms as T
from dataset import MovieGraph
from model import Model
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


MODEL_PATH = osp.join(osp.dirname(osp.realpath(__file__)), 'models/sageconv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')
dataset = MovieGraph(path, model_name='bipartite_gnn')
data = dataset[0].to(device)

# Add user node features for message passing:
data['user'].x = torch.eye(data['user'].num_nodes, device=device)
del data['user'].num_nodes

# Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
data = T.ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.'''

model = Model(data, hidden_channels=32).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

import argparse
import getpass

parser = argparse.ArgumentParser()
parser.add_argument('-u', metavar="user", default='1', help="UserId for which we want to make a recommendation")
parser.add_argument('-n', metavar="movies", default='1', help="Number of movies we want to recommend")
parsed_args = parser.parse_args()
args = vars(parsed_args)

USERID = int(args['u'])
NUM_MOVIES = int(args['n'])

# Loading data
import pandas as pd
df_movies = pd.read_csv('../data/raw/ml-100K/movies.csv', sep="|")
df_user = pd.read_csv('../data/raw/ml-100K/user.csv')
df_ratings = pd.read_csv('../data/raw/ml-100K/ratings.csv')

movie_mapping = {idx: i for i, idx in enumerate(df_movies.movieId.unique())}
user_mapping = {idx: i for i, idx in enumerate(df_user.userId.unique())}
df_movies["movieId"] = [movie_mapping[idx] for idx in df_movies["movieId"]]
df_ratings["movieId"] = [movie_mapping[idx] for idx in df_ratings["movieId"]]
df_ratings["userId"] = [movie_mapping[idx] for idx in df_ratings["userId"]]

real_user = user_mapping[USERID]
len_movies = len(data['movie'].x)

already_seen = list(df_ratings[df_ratings["userId"]==real_user]["movieId"])

row = torch.tensor([real_user] * len_movies)
col = torch.arange(len_movies)

edge_label_index = torch.stack([row, col], dim=0)
pred = model(data.x_dict, data.edge_index_dict, edge_label_index)
pred = pred.clamp(min=0, max=5)
pred[already_seen] = 0

idx_max = torch.topk(pred, NUM_MOVIES).indices

print('Recommended movies for userId ' + str(USERID))
for i in idx_max:
    print(df_movies.loc[int(i)]["movie title"])