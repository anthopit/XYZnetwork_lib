from classes import transportnetwork as tn
import networkx as nx
from preprocessing import Preprocessing as pp

# Create a new transport network
G = pp.create_network_from_trailway("../data/Railway Data_JL.xlsx")
TN = tn.TransportNetwork(G, pos_argument=["lon", "lat"], time_arguments=["dep_time", "arr_time"])
print(TN)
print(TN.get_max_time(), TN.get_min_time())