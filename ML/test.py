import preprocessing.Preprocessing as pp
import classes.transportnetwork as tn
from visualisation.visualisation import *
from ML.embedding import *
from clustering.cluster import *

G = pp.create_network_from_trailway('../data/Railway Data_JL.xlsx')
TN = tn.TransportNetwork(G, pos_argument=["lon", "lat"])

# node2vec = Node2Vec()
# emb_df = node2vec.get_embedding(TN.get_higher_complexity())
# node2vec.plot_embedding(TN.get_higher_complexity())

gw = GraphWave()
emb_df = gw.get_embedding_df(TN.get_higher_complexity())
gw.plot_embedding(TN.get_higher_complexity())

clusters_dct = get_clusters(TN.get_higher_complexity(), type='spectral', embedding=emb_df, k=20)
plot_clusters_embedding(emb_df, clusters_dct)
map_clusters(TN, clusters_dct)
