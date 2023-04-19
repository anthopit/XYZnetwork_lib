from classes import transportnetwork as tn
from preprocessing import Preprocessing as pp
from robustness_analysis.robustness import *
from visualisation.visualisation import *
from node2vec import Node2Vec as n2v
import pandas as pd
from characterisation.distance import *
from visualisation.visualisation import *
from sklearn.preprocessing import MinMaxScaler

# Create a new transport network
G = pp.create_network_from_trailway("../../../data/Railway Data_JL.xlsx")
TN = tn.TransportNetwork(G, pos_argument=['lon', 'lat'], time_arguments=["dep_time", "arr_time"], nodes_weight_argument="lat", edges_weight_argument="train_max_speed", distance_argument="distance")
G2 = pp.create_network_from_GTFS("../../../data/gtfs")
TN2 = tn.TransportNetwork(G2, pos_argument=['lon', 'lat'])
# G3 = pp.create_network_from_edges("../../../data/road-euroroad.edges")
# TN = tn.TransportNetwork(G3)

map_network(TN)
map_network(TN2)

# graph = TN.get_higher_complexity()

# Compute the detour for each edge and add it to their attributes
# euclidian_distance, real_distance, detour = compute_distances_analysis(TN, data=True)

# Normalize the euclidian distance and minus 1
# weigth = {k: v / max(euclidian_distance.values()) for k, v in euclidian_distance.items()}
# weigth = {k: max(euclidian_distance.values()) - v for k, v in euclidian_distance.items()}
#
# # Create random weights
# weigth = {}
# for edge in list(graph.edges):
#     weigth[edge] = random.random()

# # Replace the 0 values by the minimum value
# print(euclidian_distance)

# # Extract values and reshape to have a single feature
# values = list(euclidian_distance.values())
# values = [[val] for val in values]
#
# # Create MinMaxScaler instance with feature_range=(0.01, 1) and fit to values
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler.fit(values)
#
# # Transform values using MinMaxScaler
# transformed_values = scaler.transform(values)
#
# # Update dictionary with transformed values
# for i, key in enumerate(euclidian_distance.keys()):
#     euclidian_distance[key] = transformed_values[i][0]
#     # Perform 1 - value to have a value between 0 and 1
#     euclidian_distance[key] = 1 - euclidian_distance[key]

# Add the euclidian distance to the graph
# nx.set_edge_attributes(graph, weigth, "weight")
#
# for edge in graph.edges(data=True):
#     print(edge)
#
#
# from sklearn.cluster import SpectralClustering
# from sklearn.cluster import KMeans
#
# WINDOW = 10 # Node2Vec fit window
# MIN_COUNT = 1 # Node2Vec min. count
# BATCH_WORDS = 4 # Node2Vec batch words
#
# g_emb_struct = n2v(
#     graph, # a graph g, where all nodes must be integers or strings
#     dimensions=64, # embedding dimensions (default: 128)
#     # walk_length=16, # number of nodes in each walk (default: 80)
#     #num_walks=100, # number of walks per node (default: 10)
#     weight_key="weight", # key in edge data for weight (default: None)
#     workers=1, # number of workers (default: 1)
#     p=1, # the probability for a random walk getting back to the prebious node (default: 1)
#     q=0.5, # the probability that a random walk can pass through a previously unseen part of the graph (default: 1)
# )
#
# mdl_struct = g_emb_struct.fit(
#     vector_size = 64,
#     window=WINDOW,
#     min_count=MIN_COUNT,
#     batch_words=BATCH_WORDS
# )
#
# emb_df = (
#     pd.DataFrame(
#         [mdl_struct.wv.get_vector(str(n)) for n in TN.graph.nodes()],
#         index = TN.graph.nodes
#     )
# )
#
# X = emb_df.values
#
# clustering = SpectralClustering(
#     n_clusters=30,
#     assign_labels='discretize',
#     random_state=21,
# ).fit(X)
#
# comm_dct = dict(zip(emb_df.index, clustering.labels_))
# comm_dct = {k: v + 1 for k, v in comm_dct.items()}
#
#
# plot_tsne_embedding(emb_df, node_cluster=comm_dct)
# plot_tsne_embedding(emb_df)
# map_weighted_network(TN, custom_node_weigth=comm_dct, edge_weigth=False, scale=2, node_size=5, discrete_color=True)