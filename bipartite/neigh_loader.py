import numpy as np
from torch_geometric.nn import MessagePassing
import torch
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


from typing import Optional, Callable, List
from torch_geometric.data import InMemoryDataset, HeteroData
import torch


class MovieGraph(InMemoryDataset):
    def __init__(self, root, small=True, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 model_name: Optional[str] = "neigh_load"):
        self.model_name = model_name
        self.is_small = small
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ""
    @property
    def processed_file_names(self) -> str:
        return f'data_{self.model_name}.pt'

    def download(self):
        pass

    def process(self):
        data = HeteroData()
        num_movies = 5
        num_users = 4
        data['movie'].x = torch.tensor([[1,0,1,1],[1,1,0,0],[0,0,0,1],[1,1,1,1],[0,1,1,0]])
        data['user'].x = torch.eye(num_users)
        data['user', 'rates', 'movie'].edge_index = torch.tensor([[0,0,1,1,2,2,2,3],[0,2,3,4,0,1,2,4]])
        data['user', 'rates', 'movie'].edge_label = torch.tensor([5,3,2,2,1,3,4,1]).to(torch.long)
        torch.save(self.collate([data]), self.processed_paths[0])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = MovieGraph(root='../data')
data = dataset[0].to(device)
data = T.ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label 

train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.15,
    num_test=0.15,
    neg_sampling_ratio=0.0,
    edge_types=[('user', 'rates', 'movie')],
    rev_edge_types=[('movie', 'rev_rates', 'user')],
)(data)

print(type(train_data[('user', 'rates', 'movie')].edge_label_index))
print(train_data[('user', 'rates', 'movie')].edge_label)
print(train_data.edge_types)
loader = LinkNeighborLoader(
    train_data, 
    num_neighbors={'user':[-1, -1, -1, -1], 'movie':[-1, -1, -1, -1, -1]},
    edge_label_index=train_data[('user', 'rates', 'movie')].edge_label_index.tolist(),
    edge_label=train_data[('user', 'rates', 'movie')].edge_label,
    batch_size=1
)

sampled_data = next(iter(loader))
print(sampled_data)
