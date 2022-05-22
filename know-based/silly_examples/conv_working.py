import numpy as np
from torch_geometric.nn import MessagePassing
import torch


x = torch.tensor(np.array([[5,3,2],[0,1,2],[0,5,1],[1,0,4]]).transpose())
edge_index = torch.tensor([[0,2,1,2],[2,0,2,1]])
edge_weights = torch.tensor([0.8, 0.8, 0.2, 0.2])

class MyConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='sum')

    def forward(self, x, edge_index, edge_weights):
        x = self.propagate(edge_index, x=x, edge_weight=edge_weights)
        return x

    def message(self, x_j, edge_weight):
        return  x_j * edge_weight.view(-1, 1)

conv = MyConv()
with torch.no_grad():
    new_x = conv(x, edge_index, edge_weights)
    print(new_x)

S = np.array([[0, 0, 0.8],[0, 0, 0.2],[0.8, 0.2, 0]])
print(np.matmul(S,x))
