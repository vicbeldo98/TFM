from typing import Optional, Callable, List
from torch_geometric.data import InMemoryDataset, HeteroData
import torch


class MovieGraph(InMemoryDataset):
    def __init__(self, root, small=True, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 model_name: Optional[str] = "bipartite_gnn"):
        self.model_name = model_name
        self.is_small = small
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        if self.is_small:
            return [
                'ml-latest-small/movies.csv',
                'ml-latest-small/ratings.csv',
                'ml-latest-small/links.csv'
                'ml-latest-small/tags.csv'
            ]
        else:
            return [
                'ml-latest/movie.csv',
                'ml-latest/rating.csv',
                'ml-latest/link.csv'
                'ml-latest/tag.csv'
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
        self.raw_paths[2] -> links.csv
        self.raw_paths[3] -> tag.csv
        '''

        import pandas as pd

        data = HeteroData()
        df_mov = pd.read_csv(self.raw_paths[0])
        df_ratings = pd.read_csv(self.raw_paths[1])

        # Drop rows with wrong id or with no category
        '''df_mov = df_mov.copy(deep=True)[df_mov.id.apply(lambda x: x.isnumeric())]
        df_mov = df_mov[df_mov.genres != '[]']
        df_mov.id = df_mov.id.astype('int64')
        meta_movies = list(df_mov.id.unique())
        df_ratings = df_ratings[df_ratings['movieId'].isin(meta_movies)]'''

        # Map movieId with indexes
        movie_mapping = {idx: i for i, idx in enumerate(df_mov.movieId)}

        genres = df_mov['genres'].str.get_dummies('|').values
        data['movie'].x = torch.from_numpy(genres).to(torch.float)

        # Save number of different users
        user_mapping = {idx: i for i, idx in enumerate(df_ratings['userId'].unique())}

        data['user'].num_nodes = len(user_mapping)

        # Edges definition
        src = [user_mapping[idx] for idx in df_ratings['userId']]
        dst = [movie_mapping[idx] for idx in df_ratings['movieId']]
        edge_index = torch.tensor([src, dst])
        rating = torch.from_numpy(df_ratings['rating'].values).to(torch.long)

        data['user', 'rates', 'movie'].edge_index = edge_index
        data['user', 'rates', 'movie'].edge_label = rating
        torch.save(self.collate([data]), self.processed_paths[0])
