import numpy as np
from torch_geometric.nn import MessagePassing
import torch


x = torch.tensor(np.array([[5,3,0,1,2],[0,0,3,0,5],[0,5,0,3,2],[1,0,5,3,5]]).transpose())
edge_index = torch.tensor([[0,1,0,3,1,3,1,2,3,4],[1,0,3,0,3,1,2,1,4,3]])
edge_weights = torch.tensor([0.8, 0.8, 0.2, 0.2,0.4,0.4,0.3,0.3,0.1,0.1])

class MyConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
        self.K = 5
        #self.weight = torch.nn.Parameter(torch.Tensor(self.K))
        #self.reset_parameters()

    def forward(self, x, edge_index):
        x = self.propagate(edge_index, x=x)
        return x


    def message(self, x_j):
        print(x_j)
        return x_j

conv = MyConv()
with torch.no_grad():
    new_x = conv(x, edge_index)
    print(new_x)

S = np.array([[0,0.8,0,0.2,0],[0.8,0,0.3,0.4,0],[0,0.3,0,0,0],[0.2,0.4,0,0,0.1],[0,0,0,0.1,0]])
print(np.matmul(S,x))
print(np.matmul(np.transpose(x),S))