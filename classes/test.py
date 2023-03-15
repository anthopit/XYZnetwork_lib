from classes import transportnetwork as tn
import networkx as nx
from preprocessing import Preprocessing as pp

# Create a new transport network
G = pp.create_network_from_trailway("../data/Railway Data_JL.xlsx")
TN = tn.TransportNetwork(G, is_spatial=True, pos_arguments=["lon", "lat"])
print(TN)
print(TN.get_min_lon(), TN.get_max_lon(), TN.get_min_lat(), TN.get_max_lat())