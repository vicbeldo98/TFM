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

        model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

        for column in df_mov.columns[-19:]:
            df_mov[column] = df_mov[column].map(lambda x: column + " " if x==1 else "")

        movie_descriptions = []
        df_mov["mov_genres"] = df_mov["unknown"] + df_mov["Action"] + df_mov["Adventure"] + df_mov["Animation"] + df_mov["Children's"] + \
                               + df_mov["Comedy"] + df_mov["Crime"] + df_mov["Documentary"] + df_mov["Drama"] + df_mov["Fantasy"] + \
                               df_mov["Film-Noir"] + df_mov["Horror"] + df_mov['Musical'] + df_mov['Mystery'] + df_mov['Romance'] + \
                               df_mov['Sci-Fi'] + df_mov['Thriller'] + df_mov['War'] + df_mov['Western']
        
        for index, row in df_mov.iterrows():
            movie_descriptions.append(row["movie title"] + " with categories: " + row["mov_genres"])
        
        print(movie_descriptions)
        with torch.no_grad():
            movie_emb = model.encode(movie_descriptions, show_progress_bar=True, convert_to_tensor=True).cpu()

        data['movie'].x = movie_emb

        gender_mapping = {"M" : "male", "F" : "female"}
        user_descriptions = []
        for index, row in df_user.iterrows():
            user_descriptions.append("I am " + str(row["age"]) + ", " + gender_mapping[row["gender"]] + " and " + row["occupation"])

        print(user_descriptions)
        with torch.no_grad():
            user_emb = model.encode(user_descriptions, show_progress_bar=True, convert_to_tensor=True).cpu()

        data['user'].x = user_emb

        # Edges definition
        src = [user_mapping[idx] for idx in df_ratings['userId']]
        dst = [movie_mapping[idx] for idx in df_ratings['movieId']]
        edge_index = torch.tensor([src, dst])
        rating = torch.from_numpy(df_ratings['rating'].values).to(torch.long)

        data['user', 'rates', 'movie'].edge_index = edge_index
        data['user', 'rates', 'movie'].edge_label = rating
        torch.save(self.collate([data]), self.processed_paths[0])
