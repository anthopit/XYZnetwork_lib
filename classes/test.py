from classes import transportnetwork as tn
from preprocessing import Preprocessing as pp

# Create a new transport network
G = pp.create_network_from_trailway("../data/Railway Data_JL.xlsx")
print(G.is_directed())
TN = tn.TransportNetwork(G)