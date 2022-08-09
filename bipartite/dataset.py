from typing import Optional, Callable, List
from torch_geometric.data import InMemoryDataset, HeteroData
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


class MovieGraph(InMemoryDataset):
    def __init__(self, root, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 model_name: Optional[str] = "bipartite_gnn"):
        self.model_name = model_name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'ml-100K/movies.csv',
            'ml-100K/ratings.csv',
            'ml-100K/user.csv',
        ]

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.model_name}.pt'

    def download(self):
        pass

    def process(self):
        '''
        self.raw_paths[0] -> movies.csv
        self.raw_paths[1] -> rating.csv
        self.raw_paths[2] -> user.csv
        '''
        data = HeteroData()
        df_mov = pd.read_csv(self.raw_paths[0], sep="|")
        df_ratings = pd.read_csv(self.raw_paths[1])
        df_user = pd.read_csv(self.raw_paths[2])

        # movie id as number 
        df_mov.movieId = df_mov.movieId.astype('int64')
        # make sure that all the rating are from known movies
        meta_movies = list(df_mov.movieId.unique())
        df_ratings = df_ratings[df_ratings['movieId'].isin(meta_movies)]

        # make sure that all the ratings are from known users
        meta_users = list(df_user.userId.unique())
        df_ratings = df_ratings[df_ratings['userId'].isin(meta_users)]

        # Map movieId with indexes
        movie_mapping = {idx: i for i, idx in enumerate(df_mov.movieId.unique())}

        # Map userId with indexes
        user_mapping = {idx: i for i, idx in enumerate(df_user.userId.unique())}
                
        data['movie'].num_nodes = len(movie_mapping)

        ages = torch.unsqueeze(torch.tensor(df_user["age"]), 1).to(torch.float)

        gender_mapping = {"M" : 0, "F" : 1}
        gender = torch.unsqueeze(torch.tensor([gender_mapping[i] for i in df_user["gender"]]), 1).to(torch.float)

        occupation = torch.tensor(np.matrix(pd.get_dummies(df_user["occupation"]))).to(torch.float)

        user_emb = torch.cat([ages, gender, occupation], dim=-1)

        data['user'].x = user_emb

        # Edges definition
        src = [user_mapping[idx] for idx in df_ratings['userId']]
        dst = [movie_mapping[idx] for idx in df_ratings['movieId']]
        edge_index = torch.tensor([src, dst])
        rating = torch.from_numpy(df_ratings['rating'].values).to(torch.long)

        data['user', 'rates', 'movie'].edge_index = edge_index
        data['user', 'rates', 'movie'].edge_label = rating
        torch.save(self.collate([data]), self.processed_paths[0])
