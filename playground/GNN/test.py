from preprocessing import Preprocessing as pp
from classes import transportnetwork as tn
from data import *
from model import *
from run import *
from visualisation.visualisation import *


# Load data
G = pp.create_network_from_trailway("../../data/Railway Data_JL.xlsx")
TN = tn.TransportNetwork(G, pos_argument=['lon', 'lat'], time_arguments=['dep_time', 'arr_time'], distance_argument='distance')

args = {
    "node_features" : ["degree_one_hot", "one_hot", "constant", "pagerank", "degree", "betweenness", "closeness", "eigenvector", "clustering", "degree", "betweenness", "closeness", "eigenvector", "clustering"], # choices are ["degree_one_hot", "one_hot", "constant", "pagerank", "degree", "betweenness", "closeness", "eigenvector", "clustering", "degree", "betweenness", "closeness", "eigenvector", "clustering"]
    "train_ratio" : 0.8,
    "val_ratio" : 0.1,

    "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model" : "gcn", # choices are ["gcn", "gin", "resgcn", "gat", "graphsage", "sgc"]
    "layers" : 1,
    "hidden_channels" : 128,
    "dim_embedding" : 32,
    "save" : "ssl_model.pth",


    "lr" : 0.001,
    "epochs" : 20,
    "batch_size" : 64,
    "num_workers" : 2,



    "loss" : "infonce", # choices are ["infonce", "jensen_shannon"]
    "augment_list" : ["edge_perturbation", "node_dropping"],
    # choices are ["edge_perturbation", "diffusion", "diffusion_with_sample",
    # "node_dropping", "random_walk_subgraph", "node_attr_mask"]
}

class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

args = AttributeDict(args)

# Create data
data = create_data_from_transport_network(TN, node_features=args.node_features, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

# Create model
model = SSL_GNN(data.num_node_features, args.layers, args.hidden_channels, args.dim_embedding,  model=args.model).to(args.device)

print(model)

# Move data to device
data = data.to(args.device)

# Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Train model

train_self_supervised(data, model, optimizer, args)

emb = get_graph_embedding(data, model)

print(emb.shape)
plot_tsne_embedding(emb)

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans

# Set the number of clusters you want to obtain
n_clusters = 4

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
for i, node in enumerate(TN.graph.nodes()):
    comm_dct[node] = clustering.labels_[i]
comm_dct = {k: v + 1 for k, v in comm_dct.items()}

print(comm_dct)

plot_tsne_embedding(emb, node_cluster=comm_dct)
map_weighted_network(TN, custom_node_weigth=comm_dct, edge_weigth=False, scale=2, node_size=5, discrete_color=True)