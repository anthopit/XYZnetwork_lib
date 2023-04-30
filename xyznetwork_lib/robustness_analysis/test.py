import classes.transportnetwork as tn
import networkx as nx
import robustness
import preprocessing.Preprocessing as pp


G = pp.create_network_from_GTFS("../data/gtfs_2")
TN = tn.TransportNetwork(G, pos_argument=["lon", "lat"])

robustness.compute_robustness_analysis(TN)
robustness.plot_robustness_analysis(TN)