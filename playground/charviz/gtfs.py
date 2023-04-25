from classes import transportnetwork as tn
import numpy as np
import networkx as nx
from preprocessing import Preprocessing as pp
from visualisation.visualisation import *

import networkx as nx
import preprocessing.Preprocessing as pp
import classes.transportnetwork as tn
from robustness_analysis.robustness import *
import pandas as pd

G = pp.create_network_from_GTFS("../../data/gtfs_3")
TN = tn.TransportNetwork(G, pos_argument=["lon", "lat"])

map_network(TN)