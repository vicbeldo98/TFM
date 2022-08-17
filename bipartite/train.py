import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from dataset import MovieGraph
import matplotlib.pyplot as plt
from model import Model
import math
torch.manual_seed(0)

EPOCHS = 200
MODEL_PATH = 'models/sageconv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = MovieGraph(root='../data')
data = dataset[0].to(device)

# Add user node features for message passing:
data['user'].x = torch.eye(data['user'].num_nodes, device=device)
del data['user'].num_nodes

# Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
data = T.ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.

# Perform a link-level split into training, validation, and test edges:
# genera tres grafos, de manera aleatoria: uno para train, otro para validación y otro para test
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.10,
    num_test=0.10,
    neg_sampling_ratio=0.0,
    edge_types=[('user', 'rates', 'movie')],
    rev_edge_types=[('movie', 'rev_rates', 'user')],
)(data)

model = Model(data, hidden_channels=32).to(device)

# Due to lazy initialization, we need to run one model step so the number of parameters can be inferred:
with torch.no_grad():
    model.encoder(train_data.x_dict, train_data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)


def train():
    model.train()
    optimizer.zero_grad()
    '''
    train_data.x_dict = diccionario que contiene los embeddings iniciales tanto de los usuarios como de las peliculas
    train_data.edge_index_dict = diccionario que contiene las conexiones entre los nodos (contiene los tipos de conexiones)
    train_data['user', 'movie'].edge_label_index = diccionario que contiene las conexiones entre los nodos (contiene los tipos de conexiones)
    '''

    pred = model(train_data.x_dict, train_data.edge_index_dict, train_data['user', 'movie'].edge_label_index)
    target = train_data['user', 'movie'].edge_label.float()
    loss = F.mse_loss(pred, target)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(part, val=False):
    model.eval()
    pred = model(part.x_dict, part.edge_index_dict, part['user', 'movie'].edge_label_index)
    pred = torch.round(pred).clamp(min=1, max=5)
    target = part['user', 'movie'].edge_label.float()
    loss = F.mse_loss(pred, target)
    if val:
        scheduler.step(loss)
    print(pred.unique(return_counts=True)[1])
    rmse = loss.sqrt()
    return float(rmse)

train_rmse_list = []
val_rmse_list = []
test_rsme_list = []

BEST_MODEL_PATH = "models/best-sageconv"
best_loss = math.inf

def main():
    for epoch in range(1, EPOCHS+1):
        loss = train()
        if loss < best_loss:
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        train_rmse = test(train_data)
        train_rmse_list.append(train_rmse)
        val_rmse = test(val_data)
        val_rmse_list.append(val_rmse)
        test_rmse = test(test_data)
        test_rsme_list.append(test_rmse)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'Epoch: {epoch:03d}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}, LR: {lr:.10f}')

    torch.save(model.state_dict(), MODEL_PATH)

    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title("Evolution of RMSE in training")
    plt.plot(range(1,EPOCHS+1), train_rmse_list, label="train rmse")
    plt.plot(range(1,EPOCHS+1), val_rmse_list, label="val rmse")
    plt.plot(range(1,EPOCHS+1), test_rsme_list, label="test rmse")
    plt.legend(loc="upper left")
    plt.show()


main()
