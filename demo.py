import preprocessing.Preprocessing as pp
import classes.transportnetwork as tn
from visualisation.visualisation import *
from characterisation.degree import *
from characterisation.centrality import *
from characterisation.clustering import *
from characterisation.distance import *
from characterisation.page_rank import *
from characterisation.path import *
from characterisation.assortativity import *
from robustness_analysis.robustness import *
from ML.embedding import *
from clustering.cluster import *
from GNN.data import *
from GNN.model import *
from GNN.run import *

G = pp.create_network_from_GTFS('data/gtfs_3')
TN = tn.TransportNetwork(G, pos_argument=["lon", "lat"], time_arguments=["departure_time", "arrival_time"], distance_argument="distance")
# G = pp.create_network_from_trailway('data/Railway Data_JL.xlsx')
# TN = tn.TransportNetwork(G, pos_argument=["lon", "lat"], time_arguments=["dep_time", "arr_time"], distance_argument="distance")

args = {
    "node_features" : ["distance"], # choices are ["degree_one_hot", "one_hot", "constant", "pagerank", "degree", "betweenness", "closeness", "eigenvector", "clustering", "position", "distance"]
    "edge_attrs" : None, # choices are ["distance", "dep_time", "arr_time"]
    "train_ratio" : 0.8,
    "val_ratio" : 0.2,

    "layers" : 2,
    "model": "gcn",  # choices are ["gcn", "gin", "gat", "sage"]

    "lr" : 0.001,
    "epochs" : 100,

    "loss" : "infonce",
    "augment_list" : ["edge_perturbation", "node_dropping"],
}



args = GNNConfig(args)

# Create data
data = create_data_from_transport_network(TN.graph, TN, args).to(args.device)

# Create model
ssl_model = SSL_GNN(data.num_node_features, args).to(args.device)

# Create the optimizer
optimizer = torch.optim.Adam(ssl_model.parameters(), lr=args.lr)

# Train model
train_self_supervised(data, ssl_model, optimizer, args)

emb = get_graph_embedding(data, ssl_model)


clusters_dct = get_clusters(TN.get_higher_complexity(), type='kmeans', embedding=emb, k=20)

plot_clusters_embedding(emb, clusters_dct)
map_clusters(TN, clusters_dct)