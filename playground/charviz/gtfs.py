from classes import transportnetwork as tn
import numpy as np
import networkx as nx
from preprocessing import Preprocessing as pp
from visualisation.visualisation import *

# Load the data
G = pp.create_network_from_GTFS("../../data/gtfs_3")
TN = tn.TransportNetwork(G, pos_argument=['lon', 'lat'], time_arguments=['dep_time', 'arr_time'])

for edge in TN.get_higher_complexity().edges:
    print(edge)

# map_network(TN)
