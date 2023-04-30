from xyznetwork_lib import preprocessing as pp, classes as tn

G = pp.create_network_from_GTFS("../../data/gtfs_3")
TN = tn.TransportNetwork(G, pos_argument=["lon", "lat"])

map_network(TN)