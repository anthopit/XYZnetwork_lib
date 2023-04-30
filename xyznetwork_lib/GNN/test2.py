from preprocessing import Preprocessing as pp
from classes import transportnetwork as tn
from run import *
import pandas as pd

G = pp.create_network_from_trailway("../data/Railway Data_JL.xlsx")
TN = tn.TransportNetwork(G, pos_argument=['lon', 'lat'], time_arguments=['dep_time', 'arr_time'], distance_argument='distance')

graph=TN.get_higher_complexity()

args = {
    "node_features" : ["degree_one_hot"], # choices are ["degree_one_hot", "one_hot", "constant", "pagerank", "degree", "betweenness", "closeness", "eigenvector", "clustering", "position", "distance"]
    "node_attrs" : None,
    "edge_attrs" : None, # choices are ["distance", "dep_time", "arr_time"]
    "train_ratio" : 0.8,
    "val_ratio" : 0.1,

    "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model" : "gat", # choices are ["gcn", "gin", "gat", "sage"]
    "layers" : 2,
    "hidden_channels" : 128,
    "dim_embedding" : 64,
    "save" : "ssl_model.pth",

    "lr" : 0.001,
    "epochs" : 200,
    "num_workers" : 4,

    "loss" : "infonce",
    "augment_list" : ["edge_perturbation", "node_dropping"],
}

class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

args = AttributeDict(args)

data = create_data_from_transport_network(graph, TN, node_features=args.node_features, edge_attrs=args.edge_attrs, train_ratio=args.train_ratio, val_ratio=args.val_ratio, num_workers=args.num_workers)

# Open a csv in dataframe
df = pd.read_csv('../playground/charviz/robustness.csv')

# Create a tensor from the dataframe
tensor = torch.tensor(df.values).float()
# Keep only the first column
tensor = tensor[:,0]

data.y = tensor

print(tensor)


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(data.num_node_features, 64, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()


def train(train_data):
    model.train()
    loss_all = 0
    train_data = train_data.to(device)
    optimizer.zero_grad()
    output = model(train_data.x, train_data.edge_index)
    loss = criterion(output[train_data.train_mask], train_data.y[train_data.train_mask])
    loss.backward()
    optimizer.step()
    loss_all += loss.item()
    return loss_all / len(train_data.y)

def validate(val_data):
    model.eval()
    loss_all = 0
    val_data = val_data.to(device)
    output = model(val_data.x, val_data.edge_index)
    loss = criterion(output[val_data.val_mask], val_data.y[val_data.val_mask])
    loss_all += loss.item()
    return loss_all / len(val_data.y)


train_data = data.subgraph(data.train_mask)
print(train_data)
print(train_data.num_nodes)

val_data = data.subgraph(data.val_mask)
print(val_data)
print(val_data.num_nodes)

# Training loop
for epoch in range(400):
    train_loss = train(train_data)
    val_loss = validate(val_data)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
