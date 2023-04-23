from preprocessing import Preprocessing as pp
from classes import transportnetwork as tn
from visualisation import *
import random

G = pp.create_network_from_trailway("../data/Railway Data_JL.xlsx")
TN = tn.TransportNetwork(G, pos_argument=['lon', 'lat'], time_arguments=['dep_time', 'arr_time'], distance_argument='distance')

map_network(TN)

weight_dict = {}
for node in TN.get_higher_complexity().nodes():
    weight_dict[node] = 4*random.random()

map_weighted_network(TN, custom_node_weigth=weight_dict, edge_weigth=False)

# map_dynamic_network(TN)