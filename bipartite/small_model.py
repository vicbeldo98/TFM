import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from pyg_dataset import MovieGraph
import matplotlib.pyplot as plt

#   TODO: IMPRIMIR GRÁFICA DEL ENTRENAMIENTO

MODEL_PATH = '../models/bipartite/sageconv_mini'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = MovieGraph(root='../data')
data = dataset[0].to(device)

# Add user node features for message passing:
data['user'].x = torch.eye(data['user'].num_nodes, device=device)
del data['user'].num_nodes

# Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
data = T.ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.

print(data)
# Perform a link-level split into training, validation, and test edges:
# genera tres grafos, de manera aleatoria: uno para train, otro para validación y otro para test
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('user', 'rates', 'movie')],
    rev_edge_types=[('movie', 'rev_rates', 'user')],
)(data)


# We have an unbalanced dataset with many labels for rating 3 and 4, and very few for 0 and 1. Therefore we use a weighted MSE loss.
# Count the frequency of each value in an array of non-negative ints: https://pytorch.org/docs/stable/generated/torch.bincount.html
weight = torch.bincount(train_data['user', 'movie'].edge_label)
# Take the maximum number of appearences of a rate and normalize all with that number
weight = weight.max() / weight


def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index  # conexiones entre usuarios y peliculas
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # Edge label index parece todas las conexiones independientemente del tipo
        z_dict = self.encoder(x_dict, edge_index_dict)
        # z_dict representa todas las features de los nodos extraídas
        return self.decoder(z_dict, edge_label_index)


model = Model(hidden_channels=32).to(device)

# Due to lazy initialization, we need to run one model step so the number of parameters can be inferred:
with torch.no_grad():
    model.encoder(train_data.x_dict, train_data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       patience=10)


def train():
    model.train()
    optimizer.zero_grad()
    '''
    train_data.x_dict = diccionario que contiene los embeddings iniciales tanto de los usuarios como de las peliculas
    train_data.edge_index_dict = diccionario que contiene las conexiones entre los nodos (contiene los tipos de conexiones)
    train_data['user', 'movie'].edge_label_index = diccionario que contiene las conexiones entre los nodos (contiene los tipos de conexiones)
    '''

    pred = model(train_data.x_dict, train_data.edge_index_dict, train_data['user', 'movie'].edge_label_index)
    target = train_data['user', 'movie'].edge_label
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward()
    scheduler.step(loss)
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict, data['user', 'movie'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['user', 'movie'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)

train_rmse_list = []
val_rmse_list = []
test_rsme_list = []

def main():
    for epoch in range(1, 51):
        loss = train()
        train_rmse = test(train_data)
        train_rmse_list.append(train_rmse)
        val_rmse = test(val_data)
        val_rmse_list.append(val_rmse)
        test_rmse = test(test_data)
        test_rsme_list.append(test_rmse)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, Val: {val_rmse:.4f}, Test: {test_rmse:.4f}, LR: {lr:.10f}')

    torch.save(model.state_dict(), MODEL_PATH)

    plt.plot(range(1,51), train_rmse_list, label="train rmse")
    plt.plot(range(1,51), val_rmse_list, label="val rmse")
    plt.plot(range(1,51), test_rsme_list, label="test rmse")
    plt.legend(loc="upper left")
    plt.show()


main()
