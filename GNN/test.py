from preprocessing import Preprocessing as pp
from classes import transportnetwork as tn
from data import *
from model import *
from run import *
from visualisation.visualisation import *
from characterisation.page_rank import *
from networkx import NetworkXNoPath
from tqdm import tqdm
import multiprocessing as mp
import torch.optim as optim
from sklearn.metrics import roc_auc_score


# Load data
# G = pp.create_network_from_trailway("../../data/Railway Data_JL.xlsx")
# TN = tn.TransportNetwork(G, pos_argument=['lon', 'lat'], time_arguments=['dep_time', 'arr_time'], distance_argument='distance')

G = pp.create_network_from_GTFS("../data/gtfs_3")
TN = tn.TransportNetwork(G, pos_argument=['lon', 'lat'], time_arguments=['dep_time', 'arr_time'])
print(TN)

# Get the bniggest connected component
graph = TN.get_higher_complexity()
# print("Number of nodes: ", graph.number_of_nodes())
# print("Number of edges: ", graph.number_of_edges())
#
# # Find connected components and convert them to a list
# connected_components = list(nx.connected_components(graph))
#
# # Find the largest connected component
# largest_connected_component = max(connected_components, key=len)
#
# # Create a subgraph for the largest connected component
# graph = graph.subgraph(largest_connected_component)
#
# print("Number of nodes: ", graph.number_of_nodes())
# print("Number of edges: ", graph.number_of_edges())


args = {
    "node_features" : ["one_hot"], # choices are ["degree_one_hot", "one_hot", "constant", "pagerank", "degree", "betweenness", "closeness", "eigenvector", "clustering", "position", "distance"]
    "edge_attrs" : ["departure_time"], # choices are ["distance", "dep_time", "arr_time"]
    "train_ratio" : 0.8,
    "val_ratio" : 0.1,

    "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "layers" : 2,
    "model": "gat",  # choices are ["gcn", "gin", "gat", "sage"]
    "hidden_dim" : 128,
    "dim_embedding" : 64,
    "save" : "ssl_model.pth",

    "lr" : 0.001,
    "epochs" : 200,
    "num_workers" : 1,

    "loss" : "infonce",
    "augment_list" : ["edge_perturbation", "node_dropping"],
}



args = GNNConfig(args)

print(args.model)


#Create data
data = create_data_from_transport_network(graph, TN, args)

print(data.x)


ssl_model = SSL_GNN(data.num_node_features, args).to(args.device)

print(ssl_model)

# Move data to device
data = data.to(args.device)

# Create the optimizer
optimizer = torch.optim.Adam(ssl_model.parameters(), lr=args.lr)

# Train model
train_self_supervised(data, ssl_model, optimizer, args)

#
#
# # Create the link prediction model
# link_pred_model = GCNLinkPrediction(ssl_model).to(args.device)
#
# # Prepare the dataset
# train_examples, train_labels, test_examples, test_labels = create_link_prediction_data(data)
#
# # Set up the optimizer and loss function
# optimizer = optim.Adam(link_pred_model.parameters(), lr=0.01)
# criterion = torch.nn.BCEWithLogitsLoss()
#
# roc_aic_scores_pretrain = []
# roc_aic_scores = []
#
# num_epochs = 100
# for epoch in range(num_epochs):
#     link_pred_model.train()
#
#     optimizer.zero_grad()
#
#     node_embeddings = link_pred_model(data.x.to(args.device), data.edge_index.to(args.device))
#     train_scores = link_pred_model.predict_link(node_embeddings, torch.tensor(train_examples, dtype=torch.long).t().to(args.device)).squeeze()
#
#     loss = criterion(train_scores, torch.tensor(train_labels, dtype=torch.float).to(args.device))
#
#     loss.backward()
#     optimizer.step()
#
#     # Evaluate the model on the test set
#     link_pred_model.eval()
#     with torch.no_grad():
#         test_scores = link_pred_model.predict_link(node_embeddings,
#                                                    torch.tensor(test_examples, dtype=torch.long).t().to(
#                                                        args.device)).squeeze()
#         test_labels_tensor = torch.tensor(test_labels, dtype=torch.float).to(args.device)
#         test_loss = criterion(test_scores, test_labels_tensor)
#
#         # Calculate the ROC AUC score
#         test_scores_cpu = test_scores.cpu().detach().numpy()
#         test_labels_cpu = test_labels_tensor.cpu().detach().numpy()
#         roc_auc = roc_auc_score(test_labels_cpu, test_scores_cpu)
#
#         print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}, Test Loss: {test_loss.item()}, ROC AUC: {roc_auc}")
#         roc_aic_scores_pretrain.append(roc_auc)
#
#
# # Set up the optimizer and loss function
# optimizer = optim.Adam(link_pred_model.parameters(), lr=0.01)
# criterion = torch.nn.BCEWithLogitsLoss()
#
# num_epochs = 100
#
# data2 = create_data_from_transport_network(TN, node_features=args.node_features, edge_attrs=args.edge_attrs, train_ratio=args.train_ratio, val_ratio=args.val_ratio, num_workers=args.num_workers)
#
# model2 = SSL_GNN(data2.num_node_features, args.layers, args.hidden_channels, args.dim_embedding,  model=args.model).to(args.device)
#
# # Create another model without pretraining
# link_pred_model_no_pretrain = GCNLinkPrediction(model2).to(args.device)
#
# # Move data to device
# data2 = data.to(args.device)
#
# # Set up the optimizer and loss function for the model without pretraining
# optimizer_no_pretrain = optim.Adam(link_pred_model_no_pretrain.parameters(), lr=0.01)
#
# # Train the model without pretraining
# num_epochs = 100
# for epoch in range(num_epochs):
#     link_pred_model_no_pretrain.train()
#
#     optimizer_no_pretrain.zero_grad()
#
#     node_embeddings_no_pretrain = link_pred_model_no_pretrain(data.x.to(args.device), data.edge_index.to(args.device))
#     train_scores_no_pretrain = link_pred_model_no_pretrain.predict_link(node_embeddings_no_pretrain, torch.tensor(train_examples, dtype=torch.long).t().to(args.device)).squeeze()
#
#     loss_no_pretrain = criterion(train_scores_no_pretrain, torch.tensor(train_labels, dtype=torch.float).to(args.device))
#     loss_no_pretrain.backward()
#     optimizer_no_pretrain.step()
#
#
#     # Evaluate the model without pretraining
#     link_pred_model_no_pretrain.eval()
#     with torch.no_grad():
#         test_scores_no_pretrain = link_pred_model_no_pretrain.predict_link(node_embeddings_no_pretrain, torch.tensor(test_examples, dtype=torch.long).t().to(args.device)).squeeze()
#         test_roc_auc_no_pretrain = roc_auc_score(test_labels, test_scores_no_pretrain.cpu().numpy())
#
#         print(f"Epoch: {epoch + 1}, Loss (No Pretrain): {loss_no_pretrain.item()}, Test ROC-AUC (No Pretrain): {test_roc_auc_no_pretrain}")
#         roc_aic_scores.append(test_roc_auc_no_pretrain)
#
# # Plot the ROC AUC scores
# plt.plot(roc_aic_scores_pretrain, label="Pretrained")
# plt.plot(roc_aic_scores, label="Not Pretrained")
# plt.legend()
# plt.show()


emb = get_graph_embedding(data, ssl_model)

print(emb.shape)
plot_tsne_embedding(emb)

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans

# Set the number of clusters you want to obtain
n_clusters = 12

# Run K-means clustering on the embeddings
# clustering = SpectralClustering(
#     n_clusters=n_clusters,
#     assign_labels='discretize',
#     random_state=0
# ).fit(emb)

clustering = KMeans(
    n_clusters=n_clusters,
    random_state=0
).fit(emb)

# Assign one cluster to each node of the netwokx network
comm_dct = {}
for i, node in enumerate(graph.nodes()):
    comm_dct[node] = clustering.labels_[i]
comm_dct = {k: v + 1 for k, v in comm_dct.items()}

print(comm_dct)

plot_tsne_embedding(emb, node_cluster=comm_dct)
map_weighted_network(TN, custom_node_weigth=comm_dct, edge_weigth=False, scale=2, node_size=5, discrete_color=True)


####################################################### Training with labels #######################################################

# G = pp.create_network_from_trailway("../../data/Railway Data_JL.xlsx")
# TN = tn.TransportNetwork(G, pos_argument=['lon', 'lat'], time_arguments=['dep_time', 'arr_time'], distance_argument='distance')
#
#
# data = create_data_from_transport_network(graph, TN, node_features=args.node_features, edge_attrs=args.edge_attrs, train_ratio=args.train_ratio, val_ratio=args.val_ratio, num_workers=args.num_workers)
#
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
#
# class GNN(torch.nn.Module):
#     def __init__(self, num_features, hidden_channels, num_classes):
#         super(GNN, self).__init__()
#         self.conv1 = GCNConv(num_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, num_classes)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GNN(num_features, hidden_channels, num_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.MSELoss()
#
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32)
#
# def train():
#     model.train()
#     loss_all = 0
#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         output = model(data.x, data.edge_index)
#         loss = criterion(output[data.train_mask], data.y[data.train_mask])
#         loss.backward()
#         optimizer.step()
#         loss_all += loss.item() * data.num_graphs
#     return loss_all / len(train_dataset)
#
# def validate():
#     model.eval()
#     loss_all = 0
#     for data in val_loader:
#         data = data.to(device)
#         output = model(data.x, data.edge_index)
#         loss