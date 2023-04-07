from classes import transportnetwork as tn
from preprocessing import Preprocessing as pp
from CharVis.degree import *

# Create a new transport network
# G = pp.create_network_from_trailway("../../../data/Railway Data_JL.xlsx")
# TN = tn.TransportNetwork(G, pos_argument=["lon", "lat"], time_arguments=["dep_time", "arr_time"], nodes_weight_argument="lat", edges_weight_argument="train_max_speed")
G2 = pp.create_network_from_GTFS("../../../data/gtfs")
TN = tn.TransportNetwork(G2)
# G3 = pp.create_network_from_edges("../../../data/road-euroroad.edges")
# TN = tn.TransportNetwork(G3)

degree = map_node_degree_analysis(TN)
print(degree)