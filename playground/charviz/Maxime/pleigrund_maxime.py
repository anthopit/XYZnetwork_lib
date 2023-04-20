import characterisation.degree
from robustness_analysis import robustness
from preprocessing import Preprocessing as pp
from classes import transportnetwork as tn
from characterisation import distance
from characterisation import DefaultCharVis as dfc
from characterisation import centrality
from visualisation import visualisation
import networkx as nx

# 0.01 for chinese ; 0.05 biarritz

# Create a new transport network
G = pp.create_network_from_trailway("../../../data/Railway Data_JL.xlsx")
TN = tn.TransportNetwork(G, pos_argument=["lon", "lat"], time_arguments=["dep_time", "arr_time"],
                         nodes_weight_argument="lat", edges_weight_argument="train_max_speed")
characterisation.centrality.compute_centrality_analysis(TN)
characterisation.centrality.map_centrality_analysis(TN)

# G2 = pp.create_network_from_GTFS("../../../data/gtfs")
# TN2 = tn.TransportNetwork(G2, pos_argument=["lon", "lat"])
# robustness.plot_robustness_analysis(TN2, precision=0.05)

# robustness.map_robustness_analysis(TN2, precision=0.05, attack_type='degree')

# characterisation.degree.map_node_degree_analysis(TN)
# characterisation.degree.plot_distribution_degree_analysis(TN)
# centrality.plot_centrality_analysis(TN)
# visualisation.map_dynamic_network(TN, spatial=True, scale=2, step=100)
# visualisation.map_network(TN)
