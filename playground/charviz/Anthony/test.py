from classes import transportnetwork as tn
from preprocessing import Preprocessing as pp
from CharVis.distance import *

# Create a new transport network
G = pp.create_network_from_trailway("../../../data/Railway Data_JL.xlsx")
TN = tn.TransportNetwork(G, pos_argument=['lon', 'lat'], time_arguments=["dep_time", "arr_time"], nodes_weight_argument="lat", edges_weight_argument="train_max_speed", distance_argument="distance")
# G2 = pp.create_network_from_GTFS("../../../data/gtfs")
# TN = tn.TransportNetwork(G2)
# G3 = pp.create_network_from_edges("../../../data/road-euroroad.edges")
# TN = tn.TransportNetwork(G3)

map_detour_analysis(TN)


# # Get all the sub-networks that are not fully connected
# sub_networks = list(nx.connected_components(TN.graph))
#
# # Filter out the fully connected sub-networks
# not_fully_connected_sub_networks = [sn for sn in sub_networks if len(sn) != len(TN.graph)]
#
# # Print the sub-networks that are not fully connected
# print("Sub-networks that are not fully connected:")
# for sn in not_fully_connected_sub_networks:
#     print(sn)