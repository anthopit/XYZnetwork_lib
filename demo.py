import preprocessing.Preprocessing as pp
import classes.transportnetwork as tn
from visualisation.visualisation import *
from characterisation.degree import *
from characterisation.centrality import *
from characterisation.clustering import *
from characterisation.distance import *
from characterisation.page_rank import *
from characterisation.path import *
from characterisation.assortativity import *
from robustness_analysis.robustness import *
from ML.embedding import *
from clustering.cluster import *
from GNN.data import *
from GNN.model import *
from GNN.run import *

G = pp.create_network_from_GTFS('data/gtfs_3')
TN = tn.TransportNetwork(G, pos_argument=["lon", "lat"], time_arguments=["departure_time", "arrival_time"], distance_argument="distance")
# G = pp.create_network_from_trailway('data/Railway Data_JL.xlsx')
# TN = tn.TransportNetwork(G, pos_argument=["lon", "lat"], time_arguments=["dep_time", "arr_time"], distance_argument="distance")


gw = Node2Vec(window_size=1, p=0.25, q=4)
emb_df = gw.get_embedding_df(TN.get_higher_complexity())
gw.plot_embedding(TN.get_higher_complexity())


clusters_dct = get_clusters(TN.get_higher_complexity(), type='spectral', embedding=emb_df, k=4)
plot_clusters_embedding(emb_df, clusters_dct)
map_clusters(TN, clusters_dct)